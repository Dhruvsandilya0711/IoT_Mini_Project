# Running the experiments on Kaggle

Everything runs from one notebook: `notebooks/ciciot2023_kaggle.ipynb`. Expect **about 3–4 hours** of run time; you can close the browser while it runs.

## One-time setup

1. Sign in at <https://www.kaggle.com>.
2. **Verify your phone number**: profile picture (top right) → **Settings** → **Phone verification**. Without it, notebooks can't use the Internet, and the notebook needs Internet to download the project code.

## Steps

1. **Download the notebook file** from GitHub:
   <https://github.com/Dhruvsandilya0711/IoT_Mini_Project/blob/main/notebooks/ciciot2023_kaggle.ipynb>
   → click the **Download raw file** button (arrow icon, top right of the file view). You get `ciciot2023_kaggle.ipynb`.
2. On Kaggle: left menu **Create** (the + button) → **New Notebook**.
3. In the notebook: **File** → **Import Notebook** → upload `ciciot2023_kaggle.ipynb` → **Import**.
4. **Add the dataset**: right panel → **Input** → **+ Add Input** (or **Add Data**) → search **`CICIoT2023`**.
   Pick a dataset whose file list shows many CSV files named like `part-00000-363d1ba3-….csv` (the full release has 169 of them, ~13 GB in total). Click **+** to add it. Add only this one dataset.
5. **Turn Internet on**: right panel → **Session options** (or **Settings**) → **Internet** → **On**.
   Leave **Accelerator** as **None** (no GPU needed).
6. Click **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**.
   This runs the whole notebook in the background. You can close the tab.
7. Check progress: your profile → **Code** → open the notebook → the version shows *Running*, then *Complete*.
8. When it's complete: open the notebook's **Output** tab (or the **Output** section of the version page) → download **`results.zip`**.
9. **Unzip `results.zip` into the repository's `results/` folder.** The report and final slides are written from it.

## If something goes wrong

- **It stops within the first few minutes**: open the version's **Logs**, copy the last ~30 lines and paste them into the chat.
- **"No CICIoT2023 files found"**: the dataset wasn't added (step 4), or it isn't the CSV version.
- **The clone fails**: Internet is off (step 5) or your phone isn't verified.
- **It runs out of time**: `results.zip` is updated after every phase, so download it anyway and upload it; whatever finished can be used.
- **Want it faster**: in the second cell, change `FRAC = 0.03` to `0.02` and `SEEDS = "0 1 2"` to `"0 1"`.
