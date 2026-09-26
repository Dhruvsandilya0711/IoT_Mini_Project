# Addressing Class Imbalance in CICIoT2023-Based Intrusion Detection via SMOTE and Feature Selection

**Dhruv Sandilya (231CS122), NITK** · IoT mini project
Base paper: E. C. P. Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors* 23(13):5941, 2023. <https://doi.org/10.3390/s23135941>

## What this project does

CICIoT2023 contains IoT network traffic from 105 devices under 33 attacks in 7 categories (DDoS, DoS, Mirai, Reconnaissance, Spoofing, Web-based, Brute Force) plus benign traffic. DDoS and DoS dominate the data, and this imbalance hurts detection of the rare categories such as Web-based and Brute Force.

| Part | What | Main output |
|------|------|-------------|
| **Reproduction** | Neto et al.'s ML benchmark with Random Forest (plus Decision Tree as a lightweight baseline) on the 34-, 8- and 2-class tasks | Baseline per-class recall/F1, compared against the paper |
| **Improvement** | Rebalance the *training set only*: SMOTE, SMOTE + undersampling, and class weights for comparison | Before/after per-class tables and recall-gain charts |
| **Extension** | Feature selection (mutual information, RF importance, RFE) to shrink the 46 features for resource-constrained IoT gateways | Performance and cost (latency, model size) against the number of features |

Plan and deadlines: [`docs/TIMELINE.md`](docs/TIMELINE.md) · Course brief: [`docs/Mini-Projects.pdf`](docs/Mini-Projects.pdf)

## Repository layout

```
src/
  labels.py             34 raw classes -> 8 categories -> 2 classes
  load.py               stream raw CSVs, keep a random fraction of rows (class ratios preserved)
  preprocess.py         clean NaN/inf, frozen stratified train/val/test split
  eda.py                class imbalance, data quality, correlations
  resample.py           none | smote | hybrid | classweight (training data only)
  train.py              train + evaluate one configuration, append to results/runs.csv
  feature_selection.py  rank features on the training split (mi | rf | rfe)
  evaluate.py           metrics, per-class tables, latency, model size
  analyze.py            aggregate runs over seeds into report tables and figures
  plots.py              figure styling
scripts/run_experiments.sh   the full experiment matrix, phase by phase
tests/                  fake-data generator + end-to-end test
data/                   raw and processed data (not committed, see data/README.md)
results/                runs.csv, per-class tables, confusion matrices, figures
docs/                   timeline, course brief, paper notes
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q          # ~20 s, runs the whole pipeline on fake data
```

Get the data as described in [`data/README.md`](data/README.md) and put the CSV files under `data/raw/`.

## Running it

```bash
# 1. Sample and split (once)
python -m src.load --raw-dir data/raw --frac 0.05           # ~2.3M rows, same class ratios as the full data
python -m src.preprocess                                    # data/processed/{train,val,test}.parquet
python -m src.eda                                           # results/eda/ (Fig. 1: class imbalance)

# 2. Week 1: reproduction
scripts/run_experiments.sh baseline

# 3. Week 2: improvement
scripts/run_experiments.sh improvement

# 4. Week 3: extension (set EXT_STRATEGY to the best Week-2 strategy)
EXT_STRATEGY=smote scripts/run_experiments.sh extension
```

Single runs, for exploring:

```bash
python -m src.train --level 8 --model rf --strategy smote --smote-target 50000 --seed 0
python -m src.feature_selection --level 8 --method mi
python -m src.train --level 8 --model rf --strategy smote --k 15 --ranking results/features/ranking_mi_8class.csv
python -m src.analyze compare --level 8 --model rf
python -m src.analyze sweep --level 8 --model rf --strategy smote --ranking ranking_mi_8class
```

Every command has `--help`. Runs already in `results/runs.csv` are skipped (use `--force` to redo one), so a batch can be resumed after an interruption.

### On Kaggle or Colab

```python
!git clone https://github.com/Dhruvsandilya0711/IoT_Mini_Project.git
%cd IoT_Mini_Project
!pip install -q -r requirements.txt
!python -m src.load --raw-dir /kaggle/input/<ciciot2023-folder> --frac 0.05
!python -m src.preprocess && python -m src.eda
!scripts/run_experiments.sh baseline
```

Kaggle sessions are temporary: download `results/` (or commit it) at the end of each session.

## Outputs

| File | Contents |
|------|----------|
| `results/runs.csv` | One row per run: settings, test and validation metrics, minority-class recall, training time, prediction latency (µs per flow), model size, tree nodes |
| `results/per_class/<run_id>.csv` | Precision, recall, F1 and support per class (validation and test) |
| `results/confusion/<run_id>.{csv,png}` | Row-normalised test confusion matrix |
| `results/tables/compare_*.csv`, `results/figures/compare_*.png` | Baseline vs rebalancing, mean ± std over seeds |
| `results/tables/sweep_*.csv`, `results/figures/sweep_*.png` | Performance and cost against the number of features |
| `results/features/ranking_*.csv` | Feature rankings |
| `results/eda/` | Class counts and imbalance ratios, data quality, feature statistics, correlations |

"Minority classes" are those under 1% of the original training data; `runs.csv` lists them for every run (`minority_classes`), and `test_minority_recall` is their mean recall.

## Methodology guard-rails

1. The test set is split off first (stratified on all 34 classes) and never resampled or used to choose anything.
2. Scaling, SMOTE, undersampling and feature ranking are fit on the training split only; settings such as k are chosen on the validation split.
3. SMOTE raises minority classes to a fixed target (default 50,000 rows) rather than to the size of DDoS, which would not fit in memory.
4. Headline metrics are macro-F1 and per-class recall. Accuracy alone is misleading here because DDoS and DoS dominate.

## References

- E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors*, 23(13):5941, 2023.
- N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, 16:321–357, 2002.
