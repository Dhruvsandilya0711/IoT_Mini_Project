# Live demo

A trained gateway model replaying real CICIoT2023 flows and raising alerts, run in a Kaggle notebook. The model is
trained on 5 of the 169 CSV files so that it is ready in minutes; its numbers therefore differ slightly from the
report, which uses the full experiments.

## 1. Prepare on Kaggle (the evening before, about 10 minutes)

Either import `notebooks/demo.ipynb` (Kaggle: **File → Import Notebook**, upload the file; do not paste its text into a
cell) or make a new notebook with the three cells below. Attach the dataset **CICIoT2023 by CongyuanXu** and turn
**Internet on**. First cell:

```
import os, shutil, subprocess
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/IoT_Mini_Project", ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1",
                "https://github.com/Dhruvsandilya0711/IoT_Mini_Project.git", "/kaggle/working/IoT_Mini_Project"], check=True)
os.chdir("/kaggle/working/IoT_Mini_Project")
!python scripts/demo_prepare.py
```

Second cell (detection quality and speed on all demo flows):

```
!python scripts/detect.py --model demo/gateway_dt.joblib --flows demo/demo_flows.csv
```

Third cell (flows arriving one at a time, with an alert for each detected attack):

```
!python scripts/detect.py --model demo/gateway_dt.joblib --flows demo/demo_flows.csv --stream 40 --classes Benign Web BruteForce Recon --rate 4
```

Then **Save Version → Save & Run All**. The saved version keeps every output, so it doubles as a backup that
needs nothing to run.

## 2. Running order (about 7 minutes)

1. **Problem** (1 min): `docs/results.html`, the "1 in 2,588" grid.
2. **Results** (2 min): the recall chart (paper vs ours), then the rebalancing chart and table.
3. **Live** (3 min): the second cell (per-class recall, µs per flow, memory), then the third cell (ALERT lines
   for Web and BruteForce flows).
4. **Close** (1 min): 20 of 46 features, the 1 MB tree, the Level 5 gateway design, limitations.
