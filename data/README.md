# Data

Nothing in this folder is committed except this file.

| Path | What goes there | Made by |
|------|-----------------|---------|
| `data/raw/` | The CICIoT2023 CSV parts (`part-00000-….csv` …), in any sub-folder layout | You (download) |
| `data/processed/sample.parquet` (+ `.json`) | Random sample of all rows, class ratios kept | `python -m src.load` |
| `data/processed/{train,val,test}.parquet` (+ `split_info.json`) | Cleaned, frozen, stratified split | `python -m src.preprocess` |

## Getting CICIoT2023

- **Official:** the CIC dataset page, <https://www.unb.ca/cic/datasets/iotdataset-2023.html> (asks you to fill in a short form). Download the CSV version, not the PCAPs.
- **Kaggle (used for this project):** [CICIoT2023 by CongyuanXu](https://www.kaggle.com/datasets/congyuanxu/ciciot2023), a mirror of the original 169 CSV files (`csv/CICIoT2023/part-…-c000.csv`). Add it as a notebook input; `scripts/kaggle_run.py` finds the files on its own. See [`docs/KAGGLE_STEPS.md`](../docs/KAGGLE_STEPS.md).

The full CSV release is ~46M rows (~13 GB). You don't need all of it: `src.load --frac 0.05` keeps a random 5% of every file (~2.3M rows), which is plenty and fits in memory.

## Trying the pipeline without the real data

```bash
python -m tests.fake_data --out-dir data/fake_raw
python -m src.load --raw-dir data/fake_raw --frac 1.0
```
