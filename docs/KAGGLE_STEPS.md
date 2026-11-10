# Running the experiments on Kaggle

Everything runs from a few lines pasted into one notebook cell (or import `notebooks/ciciot2023_kaggle.ipynb`). With the default sample (about 1.4 million flows) the full run takes **about 3 hours** on a Kaggle CPU; you can close the browser while it runs. The script stops starting new runs after 11 hours, so it always finishes within Kaggle's 12-hour limit with `results.zip` saved.

## One-time setup

1. Sign in at <https://www.kaggle.com>.
2. **Verify your phone number**: profile picture (top right) → **Settings** → **Phone verification**. Without it, notebooks can't use the Internet, and the run needs Internet to download the project code.

## Steps

1. On Kaggle: left menu **Create** (the + button) → **New Notebook**.
2. **Add the dataset**: right panel → **Input** → **+ Add Input** → choose the **Datasets** filter (not *Notebooks*) → search **`CICIoT2023`** → add **CICIoT2023 by CongyuanXu** (`kaggle.com/datasets/congyuanxu/ciciot2023`, ~7 GB zipped).
   It holds the original CSV release: 169 files `csv/CICIoT2023/part-00000-363d1ba3-…-c000.csv` (~13 GB unzipped) and the paper as a PDF. Any other copy with those 169 `part-…csv` files works too (e.g. `unb-cic-iot-dataset`).
   Remove any other input (the **×** next to it), especially other users' notebook outputs: those files have been modified.
3. **Turn Internet on**: right panel → **Session options** → **Internet** → **On**. Leave **Accelerator** as **None**.
4. Delete anything in the first code cell and paste exactly these lines:

   ```
   import os, shutil, subprocess
   os.chdir("/kaggle/working")
   shutil.rmtree("/kaggle/working/IoT_Mini_Project", ignore_errors=True)
   subprocess.run(["git", "clone", "-q", "--depth", "1",
                   "https://github.com/Dhruvsandilya0711/IoT_Mini_Project.git", "/kaggle/working/IoT_Mini_Project"], check=True)
   os.chdir("/kaggle/working/IoT_Mini_Project")
   !python scripts/kaggle_run.py
   ```

   It is safe to run again in the same session: it moves to `/kaggle/working` and removes an earlier copy first.

5. Click **Save Version** (top right) → **Save & Run All (Commit)** → **Save**. It runs in the background; you can close the tab.
6. Check progress: your profile → **Code** → open the notebook → the version shows *Running*, then *Complete*.
7. When it's complete: open the version's **Output** section → download **`results.zip`**.
8. Unzip `results.zip` into the repository's `results/` folder, then run `python scripts/report_figures.py` and `python report/build_report.py` to refresh the figures and the report.

## If something goes wrong

- **`NameError: name 'null' is not defined`**: the notebook file's text was pasted into a cell. Use the lines above instead.
- **It stops within the first few minutes**: open the version's **Logs**; the last ~30 lines show the error.
- **"No CICIoT2023 files found"**: the dataset wasn't added (step 2), or it isn't the CSV version.
- **The log says `Using 1 files` or `does not look like the full CICIoT2023 CSV release`**: the input is a notebook's output (a re-shared sample), not the dataset itself. The run still works on it, but with less data. Adding another user's *notebook* as input gives you that notebook's output files, not the dataset it read; to get the 169 files, add the *dataset* (Datasets filter).
- **`getcwd: cannot access parent directories` / `could not create work tree dir`**: the cell deleted the folder the notebook was in. Use the cell above, which moves to `/kaggle/working` first.
- **The clone fails**: Internet is off (step 3) or your phone isn't verified.
- **It runs out of time**: `results.zip` is updated after every phase, so download it anyway; it holds every finished phase.
- **Want it faster**: change the last line to `!python scripts/kaggle_run.py --frac 0.02 --seeds "0 1"` (roughly half the time).
- **Running it a second time** (e.g. after the code was updated): use **Save Version** again on the same notebook. It always downloads the latest code.
