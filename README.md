# The 2026 NeuroGolf Championship

Kaggle competition workspace for `neurogolf-2026`.

Competition: https://www.kaggle.com/competitions/neurogolf-2026

## Layout

- `data/raw/`: immutable Kaggle downloads
- `data/interim/`: temporary preprocessing outputs
- `data/processed/`: model-ready datasets
- `data/folds/`: saved validation split files
- `src/neurogolf/`: reusable package code
- `scripts/`: command line entrypoints
- `experiments/`: one directory per experiment
- `outputs/`: logs, models, figures, predictions, blends
- `reports/`: analysis notes and solution writeups
- `neurogolf_kaggle_obsidian/`: Obsidian vault for decisions and experiment notes

## First Steps

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh
```

Raw data should remain unchanged after download.
