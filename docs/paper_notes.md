# Notes on Neto et al. (2023): fill in on Day 1

E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors* 23(13):5941, 2023. <https://doi.org/10.3390/s23135941>

Fill every blank from the paper itself (cite the section, table or figure number). Don't copy numbers from other sources.

## Dataset

| Item | From the paper | Section / table |
|------|----------------|-----------------|
| Number of IoT devices | 105 | |
| Attacks / categories | 33 attacks, 7 categories | |
| How features were extracted (tool, window size) | | |
| Number of features | | |
| Total rows in the CSV release | | |
| Class distribution reported? | | |

## Their ML setup

| Item | From the paper | Section |
|------|----------------|---------|
| Models evaluated | | |
| Train/test split | | |
| Scaling / preprocessing | | |
| Hyperparameters | | |
| Metrics (and macro vs weighted averaging) | | |
| Did they use the full dataset or a subset? | | |

## Reported results (targets for the reproduction)

Copy the paper's numbers for each task. Mark the models we reproduce (RF; DT is our addition if the paper doesn't have it).

| Task | Model | Accuracy | Precision | Recall | F1 | Table |
|------|-------|----------|-----------|--------|----|-------|
| 2-class | RF | | | | | |
| 8-class | RF | | | | | |
| 34-class | RF | | | | | |
| | | | | | | |

## Differences between our setup and theirs

List everything that could explain a gap in the numbers (subset size, split, hyperparameters, preprocessing, metric averaging).

-

## Points to quote in the report

- What the paper says about class imbalance or weak classes:
- Limitations the authors mention:
- Future work they suggest (useful to justify the improvement and extension):
