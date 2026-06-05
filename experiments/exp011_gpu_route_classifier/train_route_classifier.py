from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp011_gpu_route_classifier"
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
BEST_MANIFEST = ROOT / "experiments" / "exp005_top_cost_rewrite_strict" / "rewrite_manifest.csv"
ROUTES = ["crop_or_resize", "same_shape_global_transform", "sparse_edit_or_object_completion"]


def pad_grid(grid: list[list[int]], size: int = 30) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.int64)
    h = min(len(grid), size)
    w = min(len(grid[0]), size)
    arr[:h, :w] = np.asarray(grid, dtype=np.int64)[:h, :w]
    return arr


def one_hot(grid: np.ndarray, colors: int = 10) -> np.ndarray:
    out = np.zeros((colors, 30, 30), dtype=np.float32)
    clipped = np.clip(grid, 0, colors - 1)
    for color in range(colors):
        out[color] = clipped == color
    return out


def task_path(task_id: int) -> pathlib.Path:
    return DATA_DIR / f"task{task_id:03d}.json"


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads(task_path(task_id).read_text(encoding="utf-8"))


def route_for_task(task: dict[str, Any]) -> str:
    examples = task["train"] + task["test"]
    shape_pairs: set[tuple[int, int, int, int]] = set()
    same_shape = 0
    diffs: list[int] = []
    for ex in examples:
        x = np.asarray(ex["input"], dtype=np.int64)
        y = np.asarray(ex["output"], dtype=np.int64)
        shape_pairs.add((x.shape[0], x.shape[1], y.shape[0], y.shape[1]))
        if x.shape == y.shape:
            same_shape += 1
            diffs.append(int(np.sum(x != y)))
    if len(shape_pairs) == 1:
        ih, iw, oh, ow = next(iter(shape_pairs))
        if (ih, iw) != (oh, ow):
            return "crop_or_resize"
    if same_shape == len(examples) and diffs and max(diffs) <= 80:
        return "sparse_edit_or_object_completion"
    if same_shape == len(examples):
        return "same_shape_global_transform"
    return "crop_or_resize"


def example_features(ex: dict[str, Any]) -> np.ndarray:
    x_raw = np.asarray(ex["input"], dtype=np.int64)
    y_raw = np.asarray(ex["output"], dtype=np.int64)
    same_shape = float(x_raw.shape == y_raw.shape)
    diff = float(np.sum(x_raw != y_raw)) / 900.0 if x_raw.shape == y_raw.shape else -1.0
    x_colors = len(set(x_raw.ravel())) / 10.0
    y_colors = len(set(y_raw.ravel())) / 10.0
    return np.asarray(
        [
            x_raw.shape[0] / 30.0,
            x_raw.shape[1] / 30.0,
            y_raw.shape[0] / 30.0,
            y_raw.shape[1] / 30.0,
            same_shape,
            diff,
            x_colors,
            y_colors,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class Sample:
    task_id: int
    split: str
    example_index: int
    route: int
    image: np.ndarray
    features: np.ndarray


def build_samples(arc_gen_sample: int) -> tuple[list[Sample], dict[int, str]]:
    samples: list[Sample] = []
    task_routes: dict[int, str] = {}
    for task_id in range(1, 401):
        task = load_task(task_id)
        route = route_for_task(task)
        task_routes[task_id] = route
        route_id = ROUTES.index(route)
        for split in ["train", "test", "arc-gen"]:
            examples = task[split][: arc_gen_sample if split == "arc-gen" else None]
            for idx, ex in enumerate(examples):
                x = one_hot(pad_grid(ex["input"]))
                y = one_hot(pad_grid(ex["output"]))
                samples.append(
                    Sample(
                        task_id=task_id,
                        split=split,
                        example_index=idx,
                        route=route_id,
                        image=np.concatenate([x, y], axis=0),
                        features=example_features(ex),
                    )
                )
    return samples, task_routes


class RouteDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample.image),
            torch.from_numpy(sample.features),
            torch.tensor(sample.route, dtype=torch.long),
            sample.task_id,
        )


class RouteNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(ROUTES)),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(image), features], dim=1))


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    train = [s for s in samples if s.task_id % 5 != 0]
    valid = [s for s in samples if s.task_id % 5 == 0]
    return train, valid


def split_for_task(task_id: int) -> str:
    return "valid" if task_id % 5 == 0 else "train"


def evaluate(model: RouteNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for image, features, target, _task_id in loader:
            image = image.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(image, features)
            loss = criterion(logits, target)
            loss_sum += float(loss.detach().cpu()) * target.numel()
            correct += int((logits.argmax(dim=1) == target).sum().detach().cpu())
            total += target.numel()
    return {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total)}


def task_predictions(model: RouteNet, samples: list[Sample], device: torch.device) -> dict[int, np.ndarray]:
    loader = DataLoader(RouteDataset(samples), batch_size=512, shuffle=False, num_workers=0)
    probs_by_task: dict[int, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for image, features, _target, task_ids in loader:
            logits = model(image.to(device), features.to(device))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            for task_id, prob in zip(task_ids.tolist(), probs):
                probs_by_task[int(task_id)].append(prob)
    return {task_id: np.mean(rows, axis=0) for task_id, rows in probs_by_task.items()}


def load_costs() -> dict[int, float]:
    costs: dict[int, float] = {}
    if not BEST_MANIFEST.exists():
        return costs
    with BEST_MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            costs[int(row["task_id"])] = float(row["new_cost"])
    return costs


def cost_rank_map(costs: dict[int, float]) -> dict[int, int]:
    ordered = sorted(costs.items(), key=lambda item: (-item[1], item[0]))
    return {task_id: rank for rank, (task_id, _cost) in enumerate(ordered, start=1)}


def local_point(cost: float) -> float:
    return max(1.0, 25.0 - float(np.log(max(1.0, cost))))


def class_metrics(task_routes: dict[int, str], predictions: dict[int, np.ndarray]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for route in ROUTES:
        tp = fp = fn = support = 0
        for task_id, teacher in task_routes.items():
            pred = ROUTES[int(predictions[task_id].argmax())]
            if teacher == route:
                support += 1
            if pred == route and teacher == route:
                tp += 1
            elif pred == route and teacher != route:
                fp += 1
            elif pred != route and teacher == route:
                fn += 1
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "route": route,
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--arc-gen-sample", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, task_routes = build_samples(args.arc_gen_sample)
    train_samples, valid_samples = split_samples(samples)
    train_loader = DataLoader(RouteDataset(train_samples), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(RouteDataset(valid_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = RouteNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for image, features, target, _task_id in train_loader:
            image = image.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image, features)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * target.numel()
            correct += int((logits.argmax(dim=1) == target).sum().detach().cpu())
            total += target.numel()
        train_metrics = {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total)}
        valid_metrics = evaluate(model, valid_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save({"model_state": model.state_dict(), "routes": ROUTES, "args": vars(args)}, EXP_DIR / "model.pt")
    predictions = task_predictions(model, samples, device)
    costs = load_costs()
    ranks = cost_rank_map(costs)
    metrics_rows = class_metrics(task_routes, predictions)

    with (EXP_DIR / "route_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task_id",
            "split",
            "teacher_route",
            "pred_route",
            "confidence",
            "cost",
            "top_cost_rank",
            "priority_gain_if_cost_900",
        ] + [f"prob_{route}" for route in ROUTES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task_id in range(1, 401):
            prob = predictions[task_id]
            pred = ROUTES[int(prob.argmax())]
            cost = costs.get(task_id, "")
            gain_if_900 = local_point(900.0) - local_point(float(cost)) if cost != "" else ""
            row = {
                "task_id": task_id,
                "split": split_for_task(task_id),
                "teacher_route": task_routes[task_id],
                "pred_route": pred,
                "confidence": float(prob.max()),
                "cost": cost,
                "top_cost_rank": ranks.get(task_id, ""),
                "priority_gain_if_cost_900": gain_if_900,
            }
            row.update({f"prob_{route}": float(prob[idx]) for idx, route in enumerate(ROUTES)})
            writer.writerow(row)

    with (EXP_DIR / "metrics_by_class.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["route", "support", "true_positive", "false_positive", "false_negative", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    with (EXP_DIR / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "split", "teacher_route", "cost", "top_cost_rank"])
        writer.writeheader()
        for task_id in range(1, 401):
            writer.writerow(
                {
                    "task_id": task_id,
                    "split": split_for_task(task_id),
                    "teacher_route": task_routes[task_id],
                    "cost": costs.get(task_id, ""),
                    "top_cost_rank": ranks.get(task_id, ""),
                }
            )

    final_valid = history[-1]["valid_accuracy"] if history else 0.0
    macro_f1 = float(np.mean([float(row["f1"]) for row in metrics_rows])) if metrics_rows else 0.0
    result = {
        "exp_id": "exp011_gpu_route_classifier",
        "date": "2026-06-05",
        "status": "trained" if device.type == "cuda" else "trained_cpu_fallback",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "routes": ROUTES,
        "arc_gen_sample": args.arc_gen_sample,
        "seed": args.seed,
        "sample_count": len(samples),
        "train_sample_count": len(train_samples),
        "valid_sample_count": len(valid_samples),
        "teacher_route_counts": dict(Counter(task_routes.values())),
        "history": history,
        "final_valid_accuracy": final_valid,
        "macro_f1": macro_f1,
        "runtime_seconds": time.time() - started,
        "model_ignored_by_git": True,
        "leakage_risk": "中: teacher labelはrule-based taxonomy由来で、classifier出力は候補生成順のみに使う。",
        "overfitting_risk": "中: task数400のためtask_id group splitで監視する。採用判定はONNX validationとcostで行う。",
        "artifacts": {
            "model": str((EXP_DIR / "model.pt").relative_to(ROOT)),
            "route_predictions": str((EXP_DIR / "route_predictions.csv").relative_to(ROOT)),
            "metrics_by_class": str((EXP_DIR / "metrics_by_class.csv").relative_to(ROOT)),
            "split_manifest": str((EXP_DIR / "split_manifest.csv").relative_to(ROOT)),
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# exp011_gpu_route_classifier notes

## 目的

GPUで軽量CNN + tabular特徴のroute classifierを学習し、top cost taskに対するtemplate候補生成順のrankingに使う。

## 結果

- status: `{result['status']}`
- device: `{result['device']}`
- torch: `{result['torch_version']}`
- samples: `{len(samples)}`
- train/valid split: task_id group split (`task_id % 5 == 0` がvalid)
- final valid accuracy: `{final_valid:.4f}`
- macro F1: `{macro_f1:.4f}`

## Risk

- leakage risk: 中。teacher labelはrule-basedで、classifierの出力は採用判定ではなく候補生成順にだけ使う。
- overfitting risk: 中。route分類は粗い補助タスクなので、最終採用はONNX validationとofficial-like costで決める。
- `model.pt` はGit管理外。再現情報は `result.json`、`route_predictions.csv`、`metrics_by_class.csv` に残す。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
