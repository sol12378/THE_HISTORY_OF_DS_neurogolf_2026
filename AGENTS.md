# AGENTS.md

You are working in the NeuroGolf Kaggle competition workspace.

## Language Rules

- Write reasoning notes, hypotheses, decisions, experiment results, interpretation, and next actions in Japanese by default.
- Keep code, file names, config keys, Kaggle official terms, and log output in English where natural.
- Obsidian notes should primarily be written in Japanese.
- Save Markdown, JSON, CSV, and Python source as UTF-8. When writing text from Python, always pass `encoding="utf-8"`.
- When checking Japanese notes from PowerShell, prefer `Get-Content -Encoding UTF8` or Python `Path.read_text(encoding="utf-8")`; plain console output may display mojibake even when the file is valid UTF-8. See `docs/ENCODING.md`.

## Always Follow

- Give each experiment one clear purpose.
- Never modify `data/raw`.
- Do not commit secrets, raw Kaggle data, processed parquet files, model artifacts, OOF files, or submissions.
- Save reproducible outputs under `experiments/expXXX_*`.
- Update `EXP_SUMMARY.md` when an experiment changes the project state.
- Update `neurogolf_kaggle_obsidian/` when competition understanding, validation, features, submissions, or decisions change.
- Prefer fast experiments that can finish in about 20 minutes before scaling up.
- Do not optimize only for public LB.
- Explicitly document leakage risk and overfitting risk.
- After submitting to Kaggle, do not idle while waiting for grading. Check submission status when useful, but use the waiting time for the next concrete experiment, audit, documentation update, or reproducibility work.
- After submitting to Kaggle, do not idle while waiting for scoring. Use the pending time to advance independent work such as selecting the next candidate group, auditing repair sources, preparing validation scripts, drafting notes/log updates, or running non-conflicting local checks. Avoid actions that depend on the pending score until the result is known.

## Multi-Agent PDCA Workflow

- Treat the main agent as an orchestrator for implementation, experiments, and operational work.
- Use worker agents where practical for investigation, implementation variants, verification, or error analysis.
- Run worker agents with low reasoning effort by default, keeping their tasks narrow and concrete.
- Review worker outputs before accepting them. Check correctness, reproducibility, leakage risk, overfitting risk, and whether the output meets the stated metric or acceptance threshold.
- If a worker result is below the required threshold, recursively send it back for correction or assign a focused follow-up worker task.
- Continue the PDCA cycle until the objective or explicitly stated acceptance criteria are met, while keeping experiments scoped and documented.
- Record worker tasks, acceptance criteria, results, and decisions in the relevant experiment notes and PDCA logs when they affect project state.

## Before Coding

- Read `EXP_SUMMARY.md`.
- Read `neurogolf_kaggle_obsidian/00_Index/Home.md`.
- Check recent best CV, best LB, and leak-risk experiments.
- State the experiment hypothesis clearly.

## After Coding

- Save `result.json`.
- Save OOF-like validation predictions when applicable.
- Save `submission.csv` only when the output is intended for Kaggle.
- Write `notes.md`.
- Update the matching note in `neurogolf_kaggle_obsidian/05_Experiments/`.
- Update `neurogolf_kaggle_obsidian/06_PDCA/Daily_Log/YYYY-MM-DD.md`.
- Record major decisions in `neurogolf_kaggle_obsidian/06_PDCA/Decision_Log.md`.
- After a Kaggle submission, update `neurogolf_kaggle_obsidian/09_Submissions/LB_Tracking.md`.
