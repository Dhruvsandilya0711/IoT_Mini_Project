# Addressing Class Imbalance in CICIoT2023-Based Intrusion Detection via SMOTE and Feature Selection

**Dhruv Sandilya (231CS122), NITK** · IoT mini project
Base paper: E. C. P. Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors* 23(13):5941, 2023. <https://doi.org/10.3390/s23135941>

## What this project does

CICIoT2023 contains IoT network traffic from 105 devices under 33 attacks in 7 categories (DDoS, DoS, Mirai, Reconnaissance, Spoofing, Web-based, Brute Force) plus benign traffic. DDoS and DoS dominate the data, and this imbalance hurts detection of the rare categories such as Web-based and Brute Force.

| Part | What | Main output |
|------|------|-------------|
| **Reproduction** | Neto et al.'s ML benchmark with Random Forest, plus Decision Tree, LightGBM and Balanced Random Forest, on the 34-, 8- and 2-class tasks | Baseline per-class recall/F1, compared against the paper |
| **Improvement** | Fix the imbalance using the *training set only*: SMOTE, Borderline-SMOTE, SMOTE + undersampling, class weights, and decision-threshold tuning on the validation split | Before/after per-class tables, recall-gain charts, bootstrap confidence intervals and McNemar tests against the baseline |
| **Extension** | Feature selection (mutual information, RF importance, RFE) to shrink the 46 features for resource-constrained IoT gateways | Performance and cost (latency, model size) against the number of features |

Course brief: [`docs/Mini-Projects.pdf`](docs/Mini-Projects.pdf)

## Results (Phase 2)

Deliverables: **[report (PDF)](report/Phase2_Report.pdf)** · [final slides](slides/Phase2_Results.pptx) · [proposal slides](slides/Phase1_Proposal.pptx) · [interactive results page](docs/results.html) (download and open in a browser) · raw results in [`results/`](results/)

Run on the full CSV release (169 files, 3% sample = 1,401,354 flows, duplicates removed), 123 training runs, 2.7 h on a Kaggle CPU.

| | Result |
|---|---|
| **Reproduction** | Random Forest accuracy 0.995 (8 classes) vs 0.994 in the paper; same blind spot (Web recall 0.16, BruteForce 0.24). The paper's Table 6 has precision and recall swapped (checked against its own confusion matrix). |
| **Improvement** | Every imbalance method beats the Random Forest baseline (paired bootstrap p < 0.05). Threshold tuning: macro-F1 0.785 → 0.851 at no size cost; SMOTE: rare-class recall 0.398 → 0.626. |
| **Extension** | 20 of 46 features keep macro-F1 (0.856) with 22% faster prediction and a 28% smaller model; a class-weighted Decision Tree reaches 0.841 in 1.1 MB. |
| **Data hygiene** | Without de-duplication, 15.9% of test rows have an exact copy in the training data. |

The report follows the 10-step IoT design methodology (purpose, process model, domain model, information model, services, IoT level, functional view, operational view, integration, application development).

## Repository layout

```
src/
  labels.py             34 raw classes -> 8 categories -> 2 classes
  load.py               stream raw CSVs, keep a random fraction of rows (class ratios preserved)
  preprocess.py         clean NaN/inf, frozen stratified train/val/test split
  eda.py                class imbalance, data quality, correlations
  resample.py           none | smote | hybrid | classweight | borderline | threshold (training data only)
  threshold.py          per-class decision weights tuned on the validation split
  train.py              train + evaluate one configuration (dt | rf | lgbm | brf), append to results/runs.csv
  feature_selection.py  rank features on the training split (mi | rf | rfe)
  evaluate.py           metrics, per-class tables, latency, model size
  stats.py              bootstrap confidence intervals and McNemar tests on saved predictions
  analyze.py            aggregate runs over seeds into report tables and figures
  plots.py              figure styling
scripts/run_experiments.sh   the full experiment matrix, phase by phase
scripts/pi_detect.py         replay flows through a saved model on a Raspberry Pi gateway
tests/                  fake-data generator + end-to-end test
data/                   raw and processed data (not committed, see data/README.md)
results/                runs.csv, per-class tables, confusion matrices, figures
docs/                   course brief, Kaggle and Raspberry Pi guides, interactive results page (results.html)
report/                 Phase 2 report: build_report.py -> report.html -> Phase2_Report.pdf, figures, diagrams (Graphviz)
slides/                 Phase 1 proposal and Phase 2 results decks
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

# 2. Reproduction
scripts/run_experiments.sh baseline

# 3. Improvement
scripts/run_experiments.sh improvement

# 4. Extension (set EXT_STRATEGY to the best improvement strategy)
EXT_STRATEGY=smote scripts/run_experiments.sh extension
```

Single runs, for exploring:

```bash
python -m src.train --level 8 --model rf --strategy smote --smote-target 50000 --seed 0
python -m src.feature_selection --level 8 --method mi
python -m src.train --level 8 --model rf --strategy smote --k 15 --ranking results/features/ranking_mi_8class.csv
python -m src.analyze compare --level 8 --model rf        # incl. significance vs baseline
python -m src.analyze leaderboard --level 8               # every model x strategy
python -m src.analyze sweep --level 8 --model rf --strategy smote --ranking ranking_mi_8class
python -m src.analyze duplicates --level 8                # scores with vs without duplicate rows
```

Every command has `--help`. Runs already in `results/runs.csv` are skipped (use `--force` to redo one), so a batch can be resumed after an interruption.

### On Kaggle (recommended)

Step-by-step guide: [`docs/KAGGLE_STEPS.md`](docs/KAGGLE_STEPS.md). In a Kaggle notebook cell, `!python scripts/kaggle_run.py` (after cloning the repo) runs every phase and packs the results into `results.zip`.

### On a Raspberry Pi (edge gateway demo)

Train the class-weighted Decision Tree on the Pi and replay flows through it with `scripts/pi_detect.py`: [`docs/RASPBERRY_PI.md`](docs/RASPBERRY_PI.md).

### By hand on Kaggle or Colab

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
| `results/predictions/<run_id>.npz` | Test labels and predictions, used for the significance tests |
| `results/tables/compare_*_significance.csv` | Each strategy vs the baseline: change in macro-F1 and minority recall with 95% bootstrap CIs, McNemar p-value |
| `results/tables/leaderboard_*.csv`, `results/figures/leaderboard_*.png` | Every model x strategy, ranked by macro-F1 |
| `results/tables/duplicates_8c.csv` | Baseline scores on data with and without exact duplicate rows |
| `results/tables/compare_*.csv`, `results/figures/compare_*.png` | Baseline vs rebalancing, mean ± std over seeds |
| `results/tables/sweep_*.csv`, `results/figures/sweep_*.png` | Performance and cost against the number of features |
| `results/features/ranking_*.csv` | Feature rankings |
| `results/eda/` | Class counts and imbalance ratios, data quality, feature statistics, correlations |

"Minority classes" are those under 1% of the original training data; `runs.csv` lists them for every run (`minority_classes`), and `test_minority_recall` is their mean recall.

## Methodology guard-rails

1. Exact duplicate rows are removed before splitting, so no test row has a copy in the training data (the `duplicates` phase measures how much such copies inflate scores). The test set is split off first (stratified on all 34 classes) and never resampled or used to choose anything.
2. Scaling, SMOTE, undersampling and feature ranking are fit on the training split only; settings such as k are chosen on the validation split.
3. SMOTE raises minority classes to a fixed target (default 50,000 rows) rather than to the size of DDoS, which would not fit in memory.
4. Headline metrics are macro-F1 and per-class recall. Accuracy alone is misleading here because DDoS and DoS dominate.
5. Every claimed improvement is checked against the baseline trained with the same seed, on the same test rows (paired bootstrap and McNemar test).

## References

- E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors*, 23(13):5941, 2023.
- N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, 16:321–357, 2002.
- H. Han, W.-Y. Wang, B.-H. Mao, "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning," *ICIC*, LNCS 3644, pp. 878–887, 2005.
- C. Chen, A. Liaw, L. Breiman, "Using Random Forest to Learn Imbalanced Data," Tech. Rep. 666, UC Berkeley, 2004.
- G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *NeurIPS*, 2017.
