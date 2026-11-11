# Indian Movie Rating Regression

Predict numeric movie ratings with categorical metadata and a random-forest regressor.

## Status and provenance

Educational project. The original notebook places this work under a CodSoft internship directory. The maintained `train.py` entry point was added during a portfolio maintenance pass with AI assistance; it is not a claim about the original internship deliverable. Original dataset redistribution rights and any tutorial source still need confirmation.

## Setup and run

Use Python 3.12. From this repository directory:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements.txt
python train.py
```

The dataset is included in this repository. Use `--data PATH` for another file with the same columns. The script prints evaluation results; no pretrained model or hosted API is supplied. `python train.py --help` lists options.

## Method and limitations

Preprocessing is fitted on training rows only. A fixed seed makes the split reproducible. Random holdout evaluation is exploratory, not a time-based forecast. Actor/director metadata is high-cardinality; compare the printed MAE with the mean-rating baseline. Votes and movie titles are excluded from predictors.

The original notebook is retained as a learning artefact. Cleared notebook outputs must be regenerated; use `train.py` for the maintained evaluation path.

## Verified run

With the included CSV, pinned direct dependencies and 100 trees: 1,584 held-out rows; MAE 0.9064, RMSE 1.1789; mean-prediction baseline MAE 1.1072 (seed 42). This random split does not establish performance on future releases.
