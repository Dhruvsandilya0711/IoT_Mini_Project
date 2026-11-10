# Running the detector on a Raspberry Pi

The Pi plays the IoT gateway from the report (IoT Level 5, detection at the edge): it trains the
class-weighted Decision Tree (1.1 MB, macro-F1 0.841 on the Kaggle run), replays recorded flows
through it and reports alerts, time per flow, memory and CPU temperature.

## What you need

- Raspberry Pi 4 or 5 with 4 GB of RAM or more, and a microSD card of 32 GB or more
- **64-bit** Raspberry Pi OS (on the 32-bit version `pyarrow` does not install)
- A laptop on the same network, to SSH into the Pi and copy files to it

## 1. Set up the Pi

1. In **Raspberry Pi Imager** choose **Raspberry Pi OS (64-bit)** (Lite is enough).
2. Under **Edit settings**: hostname `iot-gateway`, a username and password, your Wi-Fi, and **SSH** on.
3. Flash the card, boot the Pi, then from the laptop:

```bash
ssh <user>@iot-gateway.local
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git python3-venv tmux
```

## 2. Code and Python packages

```bash
git clone https://github.com/Dhruvsandilya0711/IoT_Mini_Project.git
cd IoT_Mini_Project
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas pyarrow scikit-learn imbalanced-learn scipy matplotlib joblib
```

LightGBM is left out on purpose; it is only needed for the LightGBM experiments.

## 3. Part of the dataset

The full 13 GB is too much for a Pi; about 5 of the 169 CSV files (roughly 1.4 million flows, the same
size as the Kaggle sample) are enough.

1. On the laptop, open `kaggle.com/datasets/congyuanxu/ciciot2023`, go to `csv/CICIoT2023/` in the
   file browser and download 5 of the `part-…csv` files.
2. Copy them to the Pi:

```bash
ssh <user>@iot-gateway.local "mkdir -p ~/IoT_Mini_Project/data/raw"
scp part-*.csv <user>@iot-gateway.local:~/IoT_Mini_Project/data/raw/
```

## 4. Prepare the data (on the Pi, inside `tmux`)

```bash
cd ~/IoT_Mini_Project && source .venv/bin/activate
python -m src.load --raw-dir data/raw --frac 1.0
python -m src.preprocess --drop-duplicates
```

Same cleaning, de-duplication and train/validation/test split as the main project.

## 5. Train the gateway model

```bash
python -m src.train --level 8 --model dt --strategy classweight --save-model --results-dir pi_results
```

`--results-dir pi_results` keeps the Pi's results apart from the ones in `results/`.

Optional, slower (several minutes; `--n-estimators 50` halves it): the Random Forest with threshold
tuning on the top 20 features.

```bash
python -m src.train --level 8 --model rf --strategy threshold --k 20 \
  --ranking results/features/ranking_mi_8class.csv --save-model --results-dir pi_results
```

## 6. Run the detector

```bash
ls pi_results/models/

# whole test split at once: recall and precision per class, time per flow, flows per second,
# peak memory, CPU temperature
python scripts/pi_detect.py --model pi_results/models/8c-dt-classweight-kall-s0.joblib

# one flow at a time, like a gateway receiving traffic: an ALERT line per detected attack,
# a "missed" line per attack that passes as benign
python scripts/pi_detect.py --model pi_results/models/8c-dt-classweight-kall-s0.joblib \
  --stream 300 --classes Benign Web BruteForce Recon --rate 20
```

Run `htop` in a second SSH window during the demo to show CPU and memory use.

## Notes

- **Live traffic.** The 46 features were computed by the dataset authors' own tool over windows of
  packets. Producing identical features from live packets would need that tool re-implemented, so the
  demo replays recorded flows, which is the usual way to evaluate an IDS.
- **Train elsewhere, deploy only the model.** Train on Kaggle or a laptop with `--save-model`, then
  copy the `.joblib` file and `data/processed/test.parquet` to the Pi. Both machines need the same
  scikit-learn version, or the model will not load.
- **Out of memory** (2 GB Pi): use 2–3 CSV files instead of 5.
- **`externally-managed-environment`** from pip: activate the virtual environment first
  (`source .venv/bin/activate`).
