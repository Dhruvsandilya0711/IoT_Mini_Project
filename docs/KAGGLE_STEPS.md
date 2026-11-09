# Running the experiments on Kaggle

Everything runs from three lines pasted into one notebook cell (or import `notebooks/ciciot2023_kaggle.ipynb`). Expect **about 7–10 hours** of run time; you can close the browser while it runs. The script stops starting new runs after 11 hours, so it always finishes within Kaggle's 12-hour limit with `results.zip` saved.

## One-time setup

1. Sign in at <https://www.kaggle.com>.
2. **Verify your phone number**: profile picture (top right) → **Settings** → **Phone verification**. Without it, notebooks can't use the Internet, and the run needs Internet to download the project code.

## Steps

1. On Kaggle: left menu **Create** (the + button) → **New Notebook**.
2. **Add the dataset**: right panel → **Input** → **+ Add Input** → choose the **Datasets** filter (not *Notebooks*) → search **`CICIoT2023`**.
   Pick a dataset whose files are named like `part-00000-363d1ba3-….csv` (the full release has 169 of them, ~13 GB). Click **+** to add it.
   Remove any other input (the **×** next to it), especially other users' notebook outputs such as "creating a smaller dataset…": those files have been modified.
3. **Turn Internet on**: right panel → **Session options** → **Internet** → **On**. Leave **Accelerator** as **None**.
4. Delete anything in the first code cell and paste exactly these three lines:

   ```
   !git clone -q --depth 1 -b main https://github.com/Dhruvsandilya0711/IoT_Mini_Project.git
   %cd IoT_Mini_Project
   !python scripts/kaggle_run.py
   ```

5. Click **Save Version** (top right) → **Save & Run All (Commit)** → **Save**. It runs in the background; you can close the tab.
6. Check progress: your profile → **Code** → open the notebook → the version shows *Running*, then *Complete*.
7. When it's complete: open the version's **Output** section → download **`results.zip`**.
8. **Unzip `results.zip` into the repository's `results/` folder.**

## If something goes wrong

- **`NameError: name 'null' is not defined`**: the notebook file's text was pasted into a cell. Use the three lines above instead.
- **It stops within the first few minutes**: open the version's **Logs**, copy the last ~30 lines and paste them into the chat.
- **"No CICIoT2023 files found"**: the dataset wasn't added (step 2), or it isn't the CSV version.
- **`Unknown CICIoT2023 labels`**: the input isn't the original dataset (someone renamed the classes). Use the original CSV release (step 2).
- **Reusing a notebook where the clone already ran**: put `!rm -rf /kaggle/working/IoT_Mini_Project` as the first line, or the clone fails because the folder exists.
- **The clone fails**: Internet is off (step 3) or your phone isn't verified.
- **It runs out of time**: `results.zip` is updated after every phase, so download it anyway and upload it.
- **Want it faster**: change the last line to `!python scripts/kaggle_run.py --frac 0.02 --seeds "0 1"` (roughly half the time).
- **Running it a second time** (e.g. after the code was updated): use **Save Version** again on the same notebook. It always downloads the latest code.
