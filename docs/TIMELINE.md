# Project Timeline — 3 Weeks

**Project:** Addressing Class Imbalance in CICIoT2023-Based Intrusion Detection via SMOTE and Feature Selection
**Author:** Dhruv Sandilya (231CS122)
**Window:** Sun 27 Sep 2026 → Sat 17 Oct 2026 (21 days, final submission on Day 21)
**Assumed effort:** ~3–4 focused hours/day on weekdays, more on weekends. Each week ends with a buffer/catch-up day.

---

## At a glance

| Week | Dates | Phase | Theme | Checkpoint at end of week |
|------|-------|-------|-------|---------------------------|
| 1 | 27 Sep – 3 Oct | **Reproduction** | Setup, data pipeline, EDA, baseline reproduction | RF + DT baseline results (8-class, 34-class, 2-class) on a frozen test set, compared against the paper |
| 2 | 4 Oct – 10 Oct | **Improvement** | SMOTE rebalancing (main contribution) | Baseline vs SMOTE per-class tables + plots; report Methodology/Results §1 drafted |
| 3 | 11 Oct – 17 Oct | **Extension** + delivery | Feature selection for lightweight deployment, final report, slides, submission | Final report, slides, clean reproducible repo |

---

## Reproduction → Improvement → Extension

The project must go beyond reproducing the paper. The three parts, and what each must show:

| Part | What it is | What counts as success | Where it appears |
|------|------------|------------------------|------------------|
| **Reproduction** | Neto et al.'s ML benchmark (RF; DT added as a lightweight baseline) on CICIoT2023 | Numbers in the same range as the paper, with every gap explained | Report §3, reproduction table |
| **Improvement** | SMOTE (and the hybrid / class-weight variants) on the training set to fix class imbalance | **Higher per-class recall and macro-F1 for the minority classes (Web-based, Brute Force, …) than the reproduced baseline, on the same untouched test set**, with the precision cost stated honestly | Report §4–§5.1, before/after per-class table, recall-gain chart |
| **Extension** | Feature selection (MI / RFE) to shrink the 46-feature input for resource-constrained IoT gateways | A reduced feature set (e.g. ≤ 15 features) that keeps most of the recall gain while cutting train time, inference latency and model size | Report §5.2, performance-vs-k and cost-vs-k plots |

**The headline claim of the report** should read like: *"Rebalancing raised Web-based recall from X to Y and Brute Force recall from X to Y (macro-F1 +Z) over the reproduced baseline; with only k of 46 features, the model keeps most of that gain at N× lower inference cost."* Every experiment below exists to fill in that sentence.

**If plain SMOTE doesn't beat the baseline** (possible — RF is already fairly robust to imbalance): a negative result for plain SMOTE is still a valid finding, but make sure a *positive* improvement is also on the table by Day 11. Use, in order: the hybrid (S2), class weights (S3), then decision-threshold tuning per class on a validation split. All of these are already in the Week 2 plan or take under a day to add.

---

## Week 1 — Reproduction: data and baseline

| Day | Date | Tasks | Output |
|-----|------|-------|--------|
| 1 | Sun 27 Sep | Read Neto et al. (2023) closely: dataset construction, the 46 features, the 34 → 8 → 2 label grouping, the ML setup (models, split, scaling, metrics). Copy the paper's reported numbers into a "target" table. **Get the data**: either download CSVs from the UNB CIC page (form) or use a Kaggle mirror inside a Kaggle Notebook (no 13 GB download, ~30 GB RAM). Set up the environment (Python 3.11, pandas, pyarrow, scikit-learn, imbalanced-learn, matplotlib, seaborn). | `notes/paper_notes.md`, target-numbers table, data accessible, `requirements.txt` |
| 2 | Mon 28 Sep | **Data loader**: read CSV parts in chunks, cast to `float32`, draw a **uniform per-class random sample** (e.g. 5–10% of every class — keeps the real imbalance ratio, which is what we're studying). Add label mappings 34-class → 8-class → 2-class. Save the sample as Parquet. | `src/load.py`, `data/sample.parquet` (gitignored) |
| 3 | Tue 29 Sep | **EDA**: class counts at 34/8/2 levels (log-scale bar chart), imbalance ratios (majority ÷ each class), NaN/inf/duplicate check, feature distributions, correlation heatmap. | `notebooks/01_eda.ipynb`, **Fig. 1: class imbalance** (motivation figure for the report) |
| 4 | Wed 30 Sep | **Preprocessing + evaluation harness**: clean inf/NaN, decide on duplicates, **stratified 80/20 split — save test indices and never touch the test set again**. Scaler fit on train only. Write one reusable `evaluate()` that logs accuracy, macro/weighted precision, recall, F1, per-class report, normalized confusion matrix, balanced accuracy, train time and inference time → CSV/JSON. | `src/preprocess.py`, `src/evaluate.py`, frozen split |
| 5 | Thu 1 Oct | **Baseline, 8-class**: Decision Tree and Random Forest (100 trees, `n_jobs=-1`, fixed seed). Also 2-class (quick). | `results/baseline_8class.csv`, confusion matrices |
| 6 | Fri 2 Oct | **Baseline, 34-class**. Compare all baselines against the paper's numbers; write down every gap and its likely cause (subset size, hyperparameters, preprocessing). Build the **per-class recall table** and name the weak classes (expect Web-based, Brute Force, possibly Spoofing/Recon). | `results/baseline_34class.csv`, reproduction comparison table |
| 7 | Sat 3 Oct | **Buffer + writing**: Report — Introduction, Related Work (use the course reading list: Khan et al. MQTT IDS, Rabbani et al. CIC IoT-DIAD 2024, the LightGBM MQTT IDS paper, Neto et al.), Dataset section, Baseline Methodology. | Report draft §1–§3 |

**✅ Checkpoint 1 (end of Week 1):** Baseline reproduced on a frozen test set, weak classes identified, Fig. 1 + reproduction table ready. *If behind: drop 34-class to Week 2's buffer day and continue with 8-class.*

---

## Week 2 — Improvement: SMOTE rebalancing (main contribution)

| Day | Date | Tasks | Output |
|-----|------|-------|--------|
| 8 | Sun 4 Oct | Read Chawla et al. (2002) SMOTE and the imbalanced-learn docs. **Design the sampling strategies** (applied to the *training set only*): **S1** SMOTE — raise minority classes to a target size via a `sampling_strategy` dict (do **not** oversample everything up to DDoS size — memory will blow up); **S2** hybrid — SMOTE on minorities + RandomUnderSampler on DDoS/DoS; **S3** `class_weight='balanced'` (algorithm-level reference, no resampling). Time SMOTE on a small slice first. | `src/resample.py`, runtime estimate |
| 9 | Mon 5 Oct | Run **S1 (SMOTE)** for DT and RF, 8-class. | `results/smote_8class.csv` |
| 10 | Tue 6 Oct | Run **S2 (hybrid)** and **S3 (class weights)**, 8-class. Start 34-class SMOTE if time allows. | `results/hybrid_8class.csv`, `results/classweight_8class.csv` |
| 11 | Wed 7 Oct | **Ablations**: minority target size (e.g. 10k / 50k / 100k), SMOTE `k_neighbors` (3 / 5). Re-run the best configs with **3 seeds** and report mean ± std. Finish 34-class SMOTE. **Gate:** if none of S1–S3 beats the baseline on minority recall, add per-class decision-threshold tuning on a validation split today. | `results/ablation_smote.csv` |
| 12 | Thu 8 Oct | **Analysis**: baseline vs S1/S2/S3 per-class recall/precision/F1 table; confusion-matrix diff; recall-gain bar chart per class. Discuss the precision trade-off on minority classes (SMOTE usually raises recall but can lower precision). | Figs. 2–4, main results table |
| 13 | Fri 9 Oct | **Write**: Methodology (SMOTE, strategies, leakage precautions) and Results §1 (rebalancing). | Report draft §4–§5.1 |
| 14 | Sat 10 Oct | **Buffer / catch-up.** Show interim results to your guide if possible. | — |

**✅ Checkpoint 2 (end of Week 2):** A before/after table showing a measurable improvement in minority-class recall and macro-F1 over the reproduced baseline (from S1, S2 or S3), with its precision cost. *This is the improvement the project is graded on — protect this week.*

---

## Week 3 — Extension: feature selection, then write-up and submission

| Day | Date | Tasks | Output |
|-----|------|-------|--------|
| 15 | Sun 11 Oct | **Feature ranking** on the training split: mutual information (`mutual_info_classif` on a ~200k-row stratified subsample — it's slow on millions of rows) and RF importance / RFE (on a subsample). Rank on the real (pre-SMOTE) training rows so the synthetic points don't bias the ranking; if you also rank on rebalanced data (as the abstract says), compare the two rankings. | `src/feature_selection.py`, ranked feature list |
| 16 | Mon 12 Oct | **Top-k sweep** with the best Week-2 configuration: k ∈ {5, 10, 15, 20, 30, 46}. Pipeline: select top-k → SMOTE on those features → train → evaluate. Log macro-F1, minority-class recall, **train time, inference latency per 1k flows, model size on disk, tree node count**. | `results/feature_sweep.csv` |
| 17 | Tue 13 Oct | Plots: performance vs k and cost vs k; pick final k. *Optional:* time inference on a Raspberry Pi to back the gateway claim. **Freeze all results.** Clean the repo: README (how to run), `requirements.txt`, `.gitignore` for data, results tables committed. | Figs. 5–6, final k, clean repo |
| 18 | Wed 14 Oct | **Write**: Results §2 (feature selection), Discussion, Limitations (subset size, no replay-attack class, single dataset), Conclusion and Future Work; finalize the Abstract with real numbers. | Full report draft |
| 19 | Thu 15 Oct | **Slides** (~10–12): problem → dataset & imbalance → baseline → SMOTE results → feature selection → takeaways. Proofread the report, check references and figure/table numbering. | Slides v1, report v2 |
| 20 | Fri 16 Oct | Rehearse the talk (timed), fix slides. **Reproducibility check**: re-run the pipeline from a clean environment and confirm the numbers match. | Final slides, verified code |
| 21 | Sat 17 Oct | **Submit** report, slides, repo link. Remaining time is buffer. | 🎯 Submission |

**✅ Checkpoint 3 (end of Week 3):** Everything submitted.

---

## Experiment matrix

| ID | Labels | Models | Training data | Features | Week |
|----|--------|--------|---------------|----------|------|
| E1 | 8-class | DT, RF | Original (imbalanced) | All 46 | 1 |
| E2 | 34-class, 2-class | DT, RF | Original | All 46 | 1 |
| E3 | 8-class (+34) | DT, RF | SMOTE (S1) | All 46 | 2 |
| E4 | 8-class | DT, RF | SMOTE + undersampling (S2) | All 46 | 2 |
| E5 | 8-class | DT, RF | Original + class weights (S3) | All 46 | 2 |
| E6 | 8-class | RF | S1/S2 ablations × 3 seeds | All 46 | 2 |
| E7 | 8-class | DT, RF | Best of S1–S3 | Top-k, k ∈ {5…46} | 3 |
| E8 *(optional)* | 8-class | Final model | — | Final k | 3 (Pi timing) |

All experiments use the **same frozen test set** (untouched real data, never resampled).

## Metrics reported everywhere

Accuracy, macro and weighted precision/recall/F1, **per-class recall and F1**, balanced accuracy, normalized confusion matrix, train time, inference latency. Macro-F1 and per-class recall are the headline metrics; accuracy alone is misleading here because DDoS/DoS dominate.

## Leakage rules (non-negotiable)

1. Split first; the test set is never resampled, scaled with its own statistics, or used for selection.
2. Scaler, SMOTE, undersampling and feature ranking are all fit on the **training split only**.
3. Hyperparameter or k choices are made on a validation split / CV within the training data, not on the test set.

## Proposed repo layout

```
data/            # raw + sampled data (gitignored)
src/             # load.py, preprocess.py, resample.py, feature_selection.py, evaluate.py, train.py
notebooks/       # 01_eda, 02_baseline, 03_smote, 04_feature_selection
results/         # CSV tables + figures
report/          # report source + final PDF
slides/
docs/            # course brief, this timeline, paper notes
```

## Risks and fallbacks

| Risk | Fallback |
|------|----------|
| Full dataset (~46M rows, ~13 GB) too big to download or fit in RAM | Kaggle Notebook with a mirrored copy; `float32`; per-class fractional sample; smaller fraction for ablations |
| SMOTE runs out of memory or is too slow | Cap minority targets with a `sampling_strategy` dict; use the hybrid (S2) so the majority shrinks |
| RF too slow on millions of rows | `n_jobs=-1`, 100 trees, cap `max_depth`; run ablations on a 10–20% slice of the sample |
| Can't match the paper's exact numbers | Expected with a subset — document the gap; the key comparison is *baseline vs SMOTE on the same split* |
| Falling behind | Drop in this order: Raspberry Pi timing → 34-class SMOTE → multi-seed runs → RFE (keep MI). Never cut Week 2's core SMOTE comparison |
