"""LLM proposer (P3-1): ask the brain LLM (Qwen3.5-35B-A3B via vLLM) for the
transformation rule behind a task's train examples, expressed as a STRUCTURED program
in the neurogolf_solver DSL (JSON). The LLM is only a PROPOSER — every proposed program
is verified deterministically on the FULL arc-gen set (neurogolf_solver.verify_program)
and adopted only through phase1.evaluate_candidate. The LLM never touches the gate.

Graceful degradation: if the vLLM endpoint is unreachable, propose_programs returns []
so the enumerative core still runs.
"""
from __future__ import annotations

import json
import re
from typing import Any

_DSL_SPEC = """You are given input/output grids of an abstract puzzle. Each grid is a 2D array of
integers 0-9 (0 is usually background). Infer the SINGLE transformation rule that maps every
input to its output, and express it as a program in this tiny DSL (a JSON list of steps applied
left-to-right):

steps (use only these):
  {"op":"geometric","kind":K}   K in identity|flip_h|flip_v|rot90|rot180|rot270|transpose|anti_transpose
  {"op":"recolor","map":{"<src>":<dst>, ...}}   replace colors (only changed colors needed)
  {"op":"crop","top":T,"left":L,"h":H,"w":W}    take a fixed sub-rectangle (only if output is a fixed crop)
  {"op":"tile","ry":RY,"rx":RX}                 repeat the grid RY times down, RX times right

Rules:
- Output ONLY a JSON object: {"program":[ ...steps... ]}. No prose, no markdown fences.
- The program must reproduce EVERY example exactly. If unsure, give your single best guess.
- Prefer the shortest program. Empty program {"program":[]} means identity.
"""


def render_grid(grid: list[list[int]]) -> str:
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def build_messages(examples: list[dict[str, Any]], max_examples: int = 4) -> list[dict[str, str]]:
    shown = examples[:max_examples]
    blocks = []
    for i, ex in enumerate(shown, 1):
        blocks.append(f"Example {i} INPUT:\n{render_grid(ex['input'])}\nExample {i} OUTPUT:\n{render_grid(ex['output'])}")
    user = _DSL_SPEC + "\n\n" + "\n\n".join(blocks) + "\n\nReturn the JSON program now."
    return [
        {"role": "system", "content": "You are a precise abstract-reasoning solver. Output only valid JSON."},
        {"role": "user", "content": user},
    ]


def _parse_programs(text: str) -> list[tuple]:
    """Extract one or more {"program":[...]} objects and convert to solver program tuples."""
    progs: list[tuple] = []
    for m in re.finditer(r"\{[^{}]*\"program\"\s*:\s*\[.*?\]\s*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        steps = obj.get("program", [])
        prog = []
        ok = True
        for st in steps:
            op = st.get("op")
            if op == "geometric":
                prog.append(("geometric", {"kind": str(st.get("kind", "identity"))}))
            elif op == "recolor":
                prog.append(("recolor", {"map": {str(k): int(v) for k, v in st.get("map", {}).items()}}))
            elif op == "crop":
                prog.append(("crop", {"top": int(st["top"]), "left": int(st["left"]),
                                       "h": int(st["h"]), "w": int(st["w"])}))
            elif op == "tile":
                prog.append(("tile", {"ry": int(st["ry"]), "rx": int(st["rx"])}))
            else:
                ok = False
                break
        if ok:
            progs.append(tuple(prog))
    return progs


def get_client(model: str = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", base_url: str = "http://localhost:8000/v1"):
    try:
        from autoresearch.llm import LLMClient
        return LLMClient(base_url=base_url, model=model, api_key="EMPTY", temperature=0.4, timeout=90.0)
    except Exception:
        return None


def propose_programs(examples: list[dict[str, Any]], client=None, samples: int = 1) -> list[tuple]:
    """Return candidate DSL programs proposed by the LLM (unverified — caller verifies)."""
    if client is None:
        client = get_client()
    if client is None:
        return []
    messages = build_messages(examples)
    out: list[tuple] = []
    seen: set[str] = set()
    for _ in range(max(1, samples)):
        try:
            text = client.chat(messages=messages)
        except Exception:
            break
        for prog in _parse_programs(text):
            if repr(prog) not in seen:
                seen.add(repr(prog))
                out.append(prog)
    return out
