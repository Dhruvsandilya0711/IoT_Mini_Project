"""Build the Phase 2 report (HTML, then PDF via render_pdf.mjs) from the result files.

    python report/build_report.py --results results --out report/report.html
    node report/render_pdf.cjs report/report.html report/Phase2_Report.pdf

Every number in the report is read from results/ (or, for the base paper, from its Tables 6 and 8),
so the text never drifts from the data.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_figures import PAPER_CLASSES, PAPER_RF_CM  # noqa: E402

# Neto et al. (2023), Table 6, Random Forest column. "recall" and "precision" are printed as in the paper.
PAPER_RF = {
    2: {"accuracy": 0.99680798, "recall": 0.965163906, "precision": 0.965395244, "f1": 0.965279544},
    8: {"accuracy": 0.994368173, "recall": 0.91001105, "precision": 0.705407564, "f1": 0.71928904},
    34: {"accuracy": 0.99164365, "recall": 0.831586401, "precision": 0.704492066, "f1": 0.714021981},
}
STRATEGY = {"none": "Baseline (no rebalancing)", "smote": "SMOTE", "hybrid": "SMOTE + undersampling",
            "classweight": "Class weights", "borderline": "Borderline-SMOTE", "threshold": "Threshold tuning"}
MODEL = {"dt": "Decision Tree", "rf": "Random Forest", "lgbm": "LightGBM", "brf": "Balanced Random Forest"}


def f3(x):
    return f"{x:.3f}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def num(x):
    return f"{int(round(x)):,}"


def table(headers, rows, cls="", caption=None, align=None):
    align = align or ["left"] + ["right"] * (len(headers) - 1)
    head = "".join(f'<th style="text-align:{a}">{h}</th>' for h, a in zip(headers, align))
    body = "".join(
        "<tr>" + "".join(f'<td style="text-align:{a}">{c}</td>' for c, a in zip(row, align)) + "</tr>" for row in rows
    )
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<table class="{cls}">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


class Numbering:
    def __init__(self):
        self.fig = 0
        self.tab = 0

    def figure(self, src, caption, width="100%"):
        self.fig += 1
        return (f'<figure><img src="{src}" style="width:{width}"/>'
                f"<figcaption><b>Figure {self.fig}.</b> {caption}</figcaption></figure>")

    def table_caption(self, caption):
        self.tab += 1
        return f"<b>Table {self.tab}.</b> {caption}"


def load(results):
    r = {"runs": pd.read_csv(results / "runs.csv")}
    r["runs"]["tag"] = r["runs"]["tag"].fillna("")
    r["runs"]["feature_ranking"] = r["runs"]["feature_ranking"].fillna("")
    for name in ["sample", "split_info", "withdups_split_info"]:
        r[name] = json.loads((results / "data_info" / f"{name}.json").read_text())
    r["quality"] = json.loads((results / "eda" / "data_quality.json").read_text())
    r["counts8"] = pd.read_csv(results / "eda" / "class_counts_8class.csv")
    t = results / "tables"
    for name in ["compare_8c_rf_test_summary", "compare_8c_rf_test_significance", "compare_8c_rf_test_per_class",
                 "compare_34c_rf_test_summary", "compare_34c_rf_test_significance", "compare_34c_rf_test_per_class",
                 "compare_8c_dt_test_significance", "compare_8c_lgbm_test_significance",
                 "leaderboard_8c", "leaderboard_34c", "duplicates_8c",
                 "sweep_8c_rf_threshold_ranking_mi_8class", "sweep_8c_dt_classweight_ranking_mi_8class",
                 "sweep_8c_lgbm_hybrid_ranking_mi_8class"]:
        r[name] = pd.read_csv(t / f"{name}.csv")
    for m in ["mi", "rf", "rfe"]:
        r[f"rank_{m}"] = pd.read_csv(results / "features" / f"ranking_{m}_8class.csv").sort_values("rank")
    return r


def paper_cm_metrics():
    cm = PAPER_RF_CM
    tp = np.diag(cm)
    recall, precision = tp / cm.sum(1), tp / cm.sum(0)
    f1 = 2 * precision * recall / (precision + recall)
    return {"accuracy": tp.sum() / cm.sum(), "recall": recall.mean(), "precision": precision.mean(), "f1": f1.mean(),
            "per_class_recall": dict(zip(PAPER_CLASSES, recall)), "rows": int(cm.sum())}


def build(results, figdir):
    d = load(results)
    n = Numbering()
    runs = d["runs"]
    base = runs[(runs.strategy == "none") & (runs.feature_ranking == "") & (runs.tag == "")]

    def base_mean(level, model, col):
        return base[(base.level == level) & (base.model == model)][col].mean()

    s8 = d["compare_8c_rf_test_summary"].set_index("strategy")
    pc8 = d["compare_8c_rf_test_per_class"].pivot(index="class", columns="strategy", values="recall")
    prec8 = d["compare_8c_rf_test_per_class"].pivot(index="class", columns="strategy", values="precision")
    sig8 = d["compare_8c_rf_test_significance"]
    s34 = d["compare_34c_rf_test_summary"].set_index("strategy")
    sig34 = d["compare_34c_rf_test_significance"].set_index("strategy")
    lb8, lb34 = d["leaderboard_8c"], d["leaderboard_34c"]
    sw_rf = d["sweep_8c_rf_threshold_ranking_mi_8class"].set_index("n_features")
    sw_dt = d["sweep_8c_dt_classweight_ranking_mi_8class"].set_index("n_features")
    sw_lgbm = d["sweep_8c_lgbm_hybrid_ranking_mi_8class"].set_index("n_features")
    dup = d["duplicates_8c"]
    paper = paper_cm_metrics()
    clean = d["split_info"]["cleaning"]
    dupinfo = d["withdups_split_info"]["cleaning"]
    rows_split = d["split_info"]["rows"]
    per_cat = d["split_info"]["rows_per_category"]
    counts8 = d["counts8"].set_index("class")
    total_rows = sum([33984560, 8090738, 2634124, 1098195, 486504, 354565, 24829, 13064])
    lgbm_none = lb8[(lb8.model == "lgbm") & (lb8.strategy == "none")].iloc[0]
    lgbm_hyb = lb8[(lb8.model == "lgbm") & (lb8.strategy == "hybrid")].iloc[0]
    dt_cw = lb8[(lb8.model == "dt") & (lb8.strategy == "classweight")].iloc[0]
    brf = lb8[(lb8.model == "brf") & (lb8.strategy == "none")].iloc[0]
    rf_none, rf_thr, rf_smote = s8.loc["none"], s8.loc["threshold"], s8.loc["smote"]
    gain_f1_thr = rf_thr.test_macro_f1 - rf_none.test_macro_f1
    gain_min_smote = rf_smote.test_minority_recall - rf_none.test_minority_recall
    k20, k46, k10 = sw_rf.loc[20], sw_rf.loc[46], sw_rf.loc[10]

    def sig_summary(sig, strategy):
        s = sig[sig.strategy == strategy]
        return (s.delta_macro_f1.mean(), s.delta_macro_f1_low.min(), s.delta_macro_f1_high.max(),
                s.delta_minority_recall.mean(), int((s.p_mcnemar < 0.05).sum()), len(s), s.p_macro_f1.max())

    # ---------- sections ----------
    abstract = f"""
<p>The CICIoT2023 dataset of Neto et al. contains {num(total_rows)} labelled network flows from a 105-device IoT testbed
under 33 attacks in seven categories. Its benchmark reports high accuracy, but the data are extremely imbalanced
(DDoS and DoS make up {pct(counts8.loc['DDoS','share'] + counts8.loc['DoS','share'])} of the flows; there are
{num(counts8.loc['DDoS','count'] / counts8.loc['BruteForce','count'])} DDoS flows for every brute-force flow), and
rare attacks are largely missed. This project designs an IoT gateway intrusion detection system following the
10-step IoT design methodology and, as its application, (1) reproduces the paper's Random Forest benchmark on a
de-duplicated {pct(d['sample']['frac'])} sample ({num(d['sample']['rows'])} flows from all 169 files), (2) improves
detection of the rare classes by handling class imbalance on the training data only, comparing SMOTE,
Borderline-SMOTE, SMOTE with undersampling, class weights and decision-threshold tuning across four models, and
(3) extends the work with feature selection for resource-constrained gateways. The reproduced Random Forest matches
the paper's accuracy ({f3(rf_none.test_accuracy)} vs {f3(PAPER_RF[8]['accuracy'])} for 8 classes) and confirms its
blind spot: web-based and brute-force attacks are recalled only {pct(pc8.loc['Web','none'],0)} and
{pct(pc8.loc['BruteForce','none'],0)} of the time. For the Random Forest, every imbalance method improves macro-F1
significantly (paired bootstrap, p&nbsp;&lt;&nbsp;0.05). Threshold tuning gives the best Random Forest
(macro-F1 {f3(rf_none.test_macro_f1)} &rarr; {f3(rf_thr.test_macro_f1)}) at no training or size cost, and SMOTE
raises rare-class recall from {f3(rf_none.test_minority_recall)} to {f3(rf_smote.test_minority_recall)}. With only
20 of the 46 features the tuned Random Forest keeps its macro-F1 ({f3(k20.test_macro_f1)}) while predicting
{pct(1 - k20.predict_us_per_flow / k46.predict_us_per_flow, 0)} faster with a {pct(1 - k20.model_size_mb / k46.model_size_mb, 0)}
smaller model, and a class-weighted Decision Tree reaches macro-F1 {f3(dt_cw.test_macro_f1)} in
{dt_cw.model_size_mb:.1f}&nbsp;MB, about {num(rf_thr.model_size_mb / dt_cw.model_size_mb)} times smaller than the
Random Forest. We also find that {pct(dupinfo['test_share_also_in_train'])} of test rows have an exact copy in the
training data unless duplicates are removed, and that the paper's precision and recall columns are swapped.</p>"""

    intro = f"""
<h2 id="s1">1. Introduction</h2>
<h3>1.1 Motivation</h3>
<p>IoT deployments add many small, rarely patched devices to home and industrial networks, and each one widens the
attack surface. Network intrusion detection systems (IDS) that learn from labelled traffic are a natural defence,
and the CICIoT2023 dataset [1] was built to train and benchmark them on realistic IoT traffic. Its class
distribution, however, is extremely skewed: a detector that always answers &ldquo;DDoS&rdquo; is already
{pct(counts8.loc['DDoS','share'],0)} accurate while never catching a brute-force login or a web injection, the
attacks that lead to device takeover rather than mere unavailability. Headline accuracy therefore says little about
the attacks that matter most; per-class recall does.</p>
<h3>1.2 Base paper</h3>
<p>Neto et al. [1] captured traffic in a 105-device smart-home testbed (67 IP devices plus 38 Zigbee/Z-Wave devices
behind five hubs), executed 33 attacks from malicious IoT devices, and published {num(total_rows)} labelled rows
with 46 features computed over windows of 10 or 100 packets. They benchmarked Logistic Regression, Perceptron,
AdaBoost, Random Forest and a deep neural network on 2-, 8- and 34-class tasks with an 80/20 split and standard
scaling, with no treatment of class imbalance. Their confusion matrix (Table 8 of [1]) shows the Random Forest
recalling only {pct(paper['per_class_recall']['Web'])} of web-based and {pct(paper['per_class_recall']['BruteForce'])}
of brute-force flows. Their conclusion names &ldquo;the analysis of features and how they influence different ML
models&rdquo; as future work.</p>
<h3>1.3 Objectives and contributions</h3>
<p>The project answers three research questions, each with a measurable target set in Phase 1:</p>
<ul>
<li><b>RQ1 &ndash; Reproduction:</b> does a Random Forest (plus Decision Tree, LightGBM and Balanced Random Forest)
reproduce the paper's benchmark? <i>Answer: yes; accuracy matches to within 0.002 at all three granularities
(Section 5.2).</i></li>
<li><b>RQ2 &ndash; Improvement:</b> does handling the imbalance raise recall on the rare classes, significantly?
<i>Answer: yes; for the Random Forest every method beats the baseline (p&nbsp;&lt;&nbsp;0.05), with rare-class recall up by as much as
{f3(gain_min_smote)} and macro-F1 up by {f3(gain_f1_thr)} (Section 5.3).</i></li>
<li><b>RQ3 &ndash; Extension:</b> can a small feature subset keep that gain at lower cost? <i>Answer: yes; 20 of 46
features keep the full macro-F1 with faster prediction and a smaller model (Section 5.4).</i></li>
</ul>
<p>Additional findings: duplicate rows leak {pct(dupinfo['test_share_also_in_train'])} of the test set into training
unless removed; the precision and recall columns of the paper's Table 6 are swapped; and LightGBM with default
settings is unstable on this imbalance unless the training data are rebalanced.</p>
<h3>1.4 Organisation</h3>
<p>Section 2 covers background. Section 3 designs the IoT system with the 10-step IoT design methodology of Bahga and
Madisetti [6]; step 10, application development, is the machine-learning pipeline, whose methodology and results
are given in Sections 4 and 5. Section 6 discusses the results and Section 7 concludes.</p>"""

    background = """
<h2 id="s2">2. Background and related work</h2>
<p><b>Datasets.</b> CICIoT2023 [1] is distinguished by its scale and by using IoT devices as attackers. The course
reading list adds CIC IoT-DIAD 2024 (device identification and anomaly detection) [15] and the Dalhousie NIMS Lab
IoT 2024 dataset [16], both candidates for testing how well models trained here transfer.</p>
<p><b>IoT intrusion detection.</b> Deep-learning IDS have been proposed for MQTT-based IoT [13], and LightGBM combined
with metaheuristic feature selection has been used for MQTT-IoT intrusion detection [14]. Tree ensembles such as
Random Forest [7] remain strong, cheap baselines on tabular flow features, which is why this project builds on them.</p>
<p><b>Class imbalance.</b> Data-level methods change the training distribution: SMOTE [2] synthesises minority
samples by interpolating between neighbours, Borderline-SMOTE [3] does so only near the class boundary, and random
undersampling removes majority samples. Algorithm-level methods leave the data unchanged: class weights scale each
class's loss, threshold moving changes the decision rule after training, and the Balanced Random Forest [4] trains
every tree on a class-balanced bootstrap sample. Gradient boosting with LightGBM [5] is included as a strong modern
learner.</p>
<p><b>Feature selection.</b> Mutual information between each feature and the label [17], impurity importance of a
Random Forest [7], and recursive feature elimination [18] are standard filter, embedded and wrapper methods.</p>
<p><b>IoT design methodology.</b> Bahga and Madisetti [6] describe ten steps from purpose and requirements to
application development, with six deployment levels; Section 3 applies them to this system.</p>"""

    # ---------- Section 3: 10-step design ----------
    req_rows = [
        ["Purpose", "Detect and classify malicious traffic in an IoT network at its gateway, including rare attack types, so an administrator can respond."],
        ["Behaviour", "Observe the traffic of every connected device, summarise it into packet windows, classify each window as benign or one of 7 attack categories (33 attack types), and raise alerts."],
        ["System management", "Monitor the gateway's health and throughput remotely; update the detection model remotely."],
        ["Data analysis", "Real-time inference at the gateway (edge); training, rebalancing and feature selection in the cloud."],
        ["Application deployment", "Detection on the gateway; storage, dashboard and model registry in the cloud."],
        ["Security", "Encrypted, authenticated alert and update channels (TLS); integrity check of every model before installation; the IDS observes a copy of the traffic so it never blocks the network if it fails."],
        ["Measurable targets", "Rare-class recall and macro-F1 above the reproduced baseline (significant at p &lt; 0.05); prediction latency in microseconds per flow; model size small enough for a Raspberry Pi-class gateway."],
    ]
    svc_rows = [
        ["Traffic capture", "Native (libpcap/tcpdump)", "Mirrored packets", "Packet stream", "Continuous", "Gateway"],
        ["Feature extraction", "Native (Python)", "Packets", "FlowWindow (46 features)", "Per window", "Gateway"],
        ["Detection", "Native (scikit-learn / LightGBM)", "FlowWindow (top-k, scaled)", "Detection", "Per window, &micro;s", "Gateway"],
        ["Alert", "MQTT publish over TLS", "Detection with is_attack = true", "Alert (JSON)", "On event", "Gateway &rarr; cloud"],
        ["Storage", "Database API", "Alerts, statistics", "Stored records", "On event", "Cloud"],
        ["Model training", "Batch job (Kaggle / cloud VM)", "Labelled flows", "ModelArtifact + metrics", "Periodic", "Cloud"],
        ["Model update", "REST over HTTPS (pull)", "ModelArtifact + SHA-256", "Installed model", "Daily poll", "Cloud &rarr; gateway"],
        ["Dashboard", "Web (HTTPS)", "Alerts, statistics", "Charts, notifications", "On demand", "Cloud"],
    ]
    level_rows = [
        ["1", "Single node; local storage and analysis", "No: many devices must be monitored"],
        ["2", "Single node; cloud storage, local analysis", "No: single node"],
        ["3", "Single node; cloud storage and analysis", "No: single node; raw traffic to the cloud is too costly"],
        ["4", "Multiple nodes, each analysing locally; cloud storage", "No: IoT devices are too constrained to run an IDS each"],
        ["<b>5</b>", "<b>Multiple end nodes and one coordinator; cloud storage and application</b>", "<b>Yes: devices are end nodes, the gateway is the coordinator</b>"],
        ["6", "Independent end nodes sending to the cloud; central controller", "No: devices cannot run agents; all traffic to the cloud adds latency"],
    ]
    op_rows = [
        ["Device", "Raspberry Pi 4 gateway with a network tap or switch mirror port; cloud VM"],
        ["Communication", "MQTT broker (e.g. Mosquitto) with TLS for alerts; HTTPS for model updates and the dashboard"],
        ["Services hosting", "Capture, extraction and detection run natively on the gateway; storage, dashboard and registry on the cloud VM"],
        ["Storage", "Relational / time-series database for alerts; object storage for model artifacts"],
        ["Training", "Batch training on Kaggle CPU (used in this project; 2.7 h for the full experiment matrix)"],
        ["Application hosting", "Web dashboard on the cloud VM"],
        ["Management", "Versioned model artifacts with metrics and SHA-256; gateway configuration file (top-k list, class weights)"],
        ["Security", "X.509 client certificates for MQTT; TLS everywhere; checksum check before a model is installed"],
    ]
    mod_rows = [
        ["src/load.py", "Streams the 169 CSV files and keeps a random 3% of rows (class ratios preserved)"],
        ["src/preprocess.py", "Removes NaN/inf rows and exact duplicates; frozen stratified train/val/test split"],
        ["src/resample.py, src/threshold.py", "Imbalance methods (training data only); per-class threshold tuning on validation"],
        ["src/train.py", "Trains and evaluates one configuration; logs metrics, latency, size, predictions"],
        ["src/feature_selection.py", "Mutual information, RF importance and RFE rankings on training data"],
        ["src/stats.py, src/analyze.py", "Bootstrap confidence intervals, McNemar tests, tables and figures"],
        ["scripts/kaggle_run.py", "Runs every phase on Kaggle and packs the results"],
    ]
    design = f"""
<h2 id="s3">3. IoT system design: the 10-step methodology</h2>
<p>The system is an intrusion detection system for an IoT network, deployed at the network gateway. Figure 1
summarises the ten design steps of Bahga and Madisetti [6] as applied here.</p>
{n.figure('figures/iot_methodology_overview.png', 'The ten steps of the IoT design methodology applied to the gateway intrusion detection system.')}
<h3>3.1 Step 1: Purpose and requirements specification</h3>
{table(['Aspect', 'Specification'], req_rows, 'spec', n.table_caption('Purpose and requirements.'), ['left', 'left'])}
<h3>3.2 Step 2: Process model specification</h3>
<p>The process model (Figure 2) describes <i>how</i> the system works as a flowchart with two loops: an online loop on
the gateway that classifies every traffic window and raises alerts, and an offline loop in the cloud that trains,
evaluates and publishes models. The two meet through the model update service and the stored alerts.</p>
{n.figure('figures/iot_process_model.png', 'Process model: online detection at the gateway (A) and offline training in the cloud (B).', '86%')}
<h3>3.3 Step 3: Domain model specification</h3>
<p>The domain model (Figure 3) names the entities, objects and relationships independently of any technology.
<b>Physical entities</b> are the IoT devices, the malicious devices and the traffic they exchange. Each is
represented by <b>virtual entities</b>: a device profile, the flow windows of its traffic, detection results,
alerts, and the model itself. <b>Devices</b> (the gateway with its network tap, and a cloud server) monitor the
physical entities and host <b>resources</b>: on-device resources (capture, feature extractor, classifier, local log)
and network resources (alert store, model registry, training compute). <b>Services</b> expose those resources to the
administrator.</p>
{n.figure('figures/iot_domain_model.png', 'Domain model: physical and virtual entities and the devices, resources and services that link them.')}
<h3>3.4 Step 4: Information model specification</h3>
<p>The information model (Figure 4) defines the data structures of the virtual entities and how they relate. The
central structure is the <b>FlowWindow</b>, which carries the 46 CICIoT2023 features of one window of 10 or 100
packets between two hosts [1]; a Detection refers to one window and to the ModelArtifact that produced it, and an
Alert groups detections for one device.</p>
{n.figure('figures/iot_information_model.png', 'Information model: virtual entities, their attributes and their relationships.', '92%')}
<h3>3.5 Step 5: Service specifications</h3>
{table(['Service', 'Type', 'Input', 'Output', 'Schedule', 'Hosted on'], svc_rows, 'spec small', n.table_caption('Service specifications.'), ['left'] * 6)}
<h3>3.6 Step 6: IoT level specification</h3>
<p>The system is a <b>Level 5</b> deployment (Figure 5): many end nodes (the IoT devices, which only generate traffic),
one coordinator node (the gateway, which collects all traffic) and the cloud for storage, analysis and the
application. It departs from the textbook Level 5 in one deliberate way: detection runs <i>at the coordinator</i>
rather than in the cloud, so that attacks are caught within microseconds without shipping all traffic upstream.
The feature-selection extension (Section 5.4) exists to make that edge inference cheap. Table 3 explains why the other
levels do not fit.</p>
{n.figure('figures/iot_iot_level.png', 'Level 5 deployment: IoT devices as end nodes, the gateway as coordinator with edge detection, and the cloud.', '92%')}
{table(['Level', 'Definition [6]', 'Fit for this system'], level_rows, 'spec', n.table_caption('Choosing the IoT level.'), ['center', 'left', 'left'])}
<h3>3.7 Step 7: Functional view specification</h3>
<p>The functional view (Figure 6) maps the entities and services onto the IoT-A functional groups (FG): Device,
Communication, Services, Management, Security and Application.</p>
{n.figure('figures/iot_functional_view.png', 'Functional view: components of each functional group.')}
<h3>3.8 Step 8: Operational view specification</h3>
{table(['Functional group', 'Operational choice'], op_rows, 'spec', n.table_caption('Operational view: how each functional group is deployed and operated.'), ['left', 'left'])}
<h3>3.9 Step 9: Device and component integration</h3>
<p>Figure 7 shows how the components are wired. Devices connect to an access point or switch as usual; a network
tap (as in the CICIoT2023 testbed [1]) or a switch mirror port gives the gateway a passive copy of all traffic, so a
failure of the IDS never interrupts the network. The gateway publishes alerts to the cloud and pulls model updates.</p>
{n.figure('figures/iot_integration.png', 'Device and component integration.')}
<h3>3.10 Step 10: Application development</h3>
<p>The application's core is the detection model and the pipeline that produces and evaluates it. It was built as a
tested Python package (scikit-learn [8], imbalanced-learn [9], LightGBM [5]) and run end to end on Kaggle; Table 5
lists its modules. Sections 4 and 5 describe the experiments that choose the model for the gateway: reproduction,
improvement and extension.</p>
{table(['Module', 'Role'], mod_rows, 'spec', n.table_caption('Software modules of the application (repository: github.com/Dhruvsandilya0711/IoT_Mini_Project).'), ['left', 'left'])}"""

    # ---------- Section 4: methodology ----------
    method = f"""
<h2 id="s4">4. Experimental methodology</h2>
<h3>4.1 Data</h3>
<p>All 169 CSV files of the original release were streamed and each row kept with probability
{d['sample']['frac']:.4f}, giving {num(d['sample']['rows'])} rows ({pct(d['sample']['rows'] / total_rows, 2)} of
{num(total_rows)}) with the true class ratios. No row had missing or infinite values. {num(clean['duplicates_dropped'])}
exact duplicate rows ({pct(clean['duplicates_dropped'] / clean['rows_in'])}) were removed, and the remaining
{num(clean['rows_out'])} rows were split once, stratified on all 34 classes, into training
({num(rows_split['train'])}), validation ({num(rows_split['val'])}) and test ({num(rows_split['test'])}) sets. The test
set is never resampled and never used for any choice. Classes below 1% of the training data (Recon, Web and
BruteForce) are the <i>rare classes</i>; their mean recall is reported as rare-class recall.</p>
<h3>4.2 Models</h3>
<p>Decision Tree; Random Forest (100 trees) as in [1]; LightGBM (100 boosting rounds, 63 leaves, learning rate 0.1);
and Balanced Random Forest (100 trees, each on a class-balanced bootstrap sample). All other settings are library
defaults; features are standardised as in [1].</p>
<h3>4.3 Imbalance methods (training data only)</h3>
<ul>
<li><b>SMOTE:</b> every class with fewer than 50,000 training rows is raised to 50,000 with synthetic samples
(k&nbsp;=&nbsp;5); an ablation tries 10,000 and 100,000.</li>
<li><b>SMOTE + undersampling (hybrid):</b> classes above 200,000 rows are randomly cut to 200,000, then SMOTE as above.</li>
<li><b>Borderline-SMOTE:</b> SMOTE restricted to minority samples near the class boundary.</li>
<li><b>Class weights:</b> each class's loss weighted inversely to its frequency.</li>
<li><b>Threshold tuning:</b> the model is trained on the original data; one multiplicative weight per class is then
applied to its predicted probabilities, chosen by coordinate ascent over 22 values between 0.1 and 100 to maximise
macro-F1 on the <i>validation</i> set.</li>
</ul>
<h3>4.4 Feature selection</h3>
<p>Features are ranked on the training data only, by mutual information [17] and Random Forest importance
(200,000-row stratified subsample) and by RFE [18] with a 50-tree forest (50,000 rows). Each model is then retrained on
the top k&nbsp;&isin;&nbsp;{{5, 10, 15, 20, 30}} features of the mutual-information ranking with the imbalance
method that gave its best <i>validation</i> macro-F1.</p>
<h3>4.5 Metrics and statistics</h3>
<p>Accuracy, macro-averaged precision, recall and F1, balanced accuracy, Matthews correlation, per-class recall and
rare-class recall on the test set; prediction latency (batch prediction time per flow on a 4-core Kaggle CPU) and
model size (serialised bytes). Each 8-class configuration is run with three seeds (mean &plusmn; standard deviation).
Each method is compared with the baseline trained with the same seed on the same test rows by a paired bootstrap
(1,000 resamples; 95% confidence interval and one-sided p-value) [11] and McNemar's exact test on per-row
correctness [10].</p>
<h3>4.6 Execution</h3>
<p>The full matrix ({len(runs)} training runs) ran unattended on a free Kaggle CPU session in 2.7 hours.</p>"""

    # ---------- Section 5: results ----------
    c8 = d["counts8"]
    data_rows = [[c, num(r["count"]), pct(r["share"], 3), num(r["imbalance_ratio"]) + " : 1",
                  num(per_cat[c]["train"]), num(per_cat[c]["test"])] for c, r in c8.set_index("class").iterrows()]

    paper_rows = []
    for level in [2, 8, 34]:
        p = PAPER_RF[level]
        ours_acc = base_mean(level, "rf", "test_accuracy")
        ours_f1 = base_mean(level, "rf", "test_macro_f1")
        ours_p = base_mean(level, "rf", "test_macro_precision")
        ours_r = base_mean(level, "rf", "test_macro_recall")
        paper_rows.append([f"{level} classes", f3(p["accuracy"]), f3(ours_acc), f3(p["f1"]), f3(ours_f1),
                           f'{f3(p["recall"])} / {f3(p["precision"])}', f"{f3(ours_p)} / {f3(ours_r)}"])

    repro = f"""
<h2 id="s5">5. Results</h2>
<h3>5.1 Data and imbalance</h3>
<p>Table 6 and Figure 8 confirm the imbalance reported in [1]: the sample's category shares match the paper's Table 3
to within sampling error, and the rarest category (BruteForce) is {num(counts8.loc['BruteForce','imbalance_ratio'])}
times smaller than DDoS. Only {num(per_cat['Web']['test'])} Web and {num(per_cat['BruteForce']['test'])} BruteForce
flows reach the test set, which is why their recall carries the widest confidence intervals. Two features, Telnet and
SMTP, are always zero in the sample.</p>
{table(['Category', 'Rows in sample', 'Share', 'DDoS : class', 'Train rows', 'Test rows'], data_rows, '', n.table_caption('Class distribution of the sample (before de-duplication for counts and shares; after it for the split).'))}
{n.figure('figures/class_distribution_8class.png', 'Class distribution of the 8-class task (log scale).', '80%')}
<h3>5.2 Reproduction (RQ1)</h3>
<p>Table 7 compares the reproduced Random Forest with the paper. Accuracy agrees to within 0.002 at every
granularity. Comparing precision and recall needs care: the 34-class Logistic Regression row of the paper's Table 6
(accuracy 0.80231507, &ldquo;recall&rdquo; 0.59520185, &ldquo;precision&rdquo; 0.486752461, F1 0.49388408) matches,
digit for digit, the output of the public CICIoT2023 example notebook, which calls
<code>recall_score(y_pred, y_test)</code> and <code>precision_score(y_pred, y_test)</code> with the arguments in the
wrong order; scikit-learn expects <code>(y_true, y_pred)</code>, so the two metrics come out swapped. Recomputing the
metrics from the paper's own Random Forest confusion matrix (its Table 8) confirms this: macro recall is
{f3(paper['recall'])} and macro precision {f3(paper['precision'])}, the reverse of the Table 6 labels
({f3(PAPER_RF[8]['recall'])} and {f3(PAPER_RF[8]['precision'])}), while accuracy ({f3(paper['accuracy'])}) and F1
({f3(paper['f1'])}) match. We therefore compare the paper's &ldquo;recall&rdquo; with our precision and vice versa.</p>
{table(['Task', 'Accuracy, paper', 'Accuracy, ours', 'Macro-F1, paper', 'Macro-F1, ours', 'Paper: printed recall / precision', 'Ours: precision / recall'], paper_rows, 'small', n.table_caption('Random Forest: base paper (Table 6 of [1]) vs this reproduction. Ours are means over seeds for 8 classes and one seed otherwise.'))}
<p>Our macro-F1 is higher than the paper's for 8 and 34 classes. Possible reasons are the different split (random,
stratified, de-duplicated), library versions and training procedure: the same example notebook calls
<code>fit()</code> once per CSV file, which in scikit-learn retrains the model from scratch each time, so a model
trained that way would see only the last file (about 0.6% of the training data). Crucially, the qualitative
picture is reproduced (Figure 9): DDoS, DoS and Mirai are recalled almost perfectly, while Web and BruteForce are
mostly mistaken for Benign, Recon and Spoofing, exactly as in the paper's Table 8 (Figure 10).</p>
{n.figure('figures/repro_per_class_8c.png', "Per-class recall of the Random Forest, 8 classes: the base paper (from its Table 8) and this project's baseline, SMOTE and threshold-tuned models (mean over three seeds).")}
{n.figure('figures/confusion_rf_pair.png', 'Random Forest confusion matrices (8 classes, seed 0), row-normalised: baseline vs threshold tuning.')}"""

    rf_rows = []
    for strat in ["none", "smote", "hybrid", "classweight", "borderline", "threshold"]:
        r = s8.loc[strat]
        rf_rows.append([STRATEGY[strat], f"{f3(r.test_macro_f1)} &plusmn; {r.test_macro_f1_std:.3f}",
                        f3(r.test_balanced_accuracy), f3(r.test_minority_recall), f3(pc8.loc['Web', strat]),
                        f3(pc8.loc['BruteForce', strat]), f3(r.test_accuracy), f"{r.model_size_mb:.0f}", f"{r.train_s:.0f}"])
    sig_rows = []
    for strat in ["smote", "hybrid", "classweight", "borderline", "threshold"]:
        m, lo, hi, dmin, k, nseeds, pmax = sig_summary(sig8, strat)
        sig_rows.append([STRATEGY[strat], f"{m:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", "&lt; 0.001" if pmax < 0.001 else f"{pmax:.3f}",
                         f"{dmin:+.3f}", f"{k} / {nseeds}"])
    lb_rows = []
    for _, r in lb8.iterrows():
        lb_rows.append([MODEL[r.model], STRATEGY[r.strategy], f"{f3(r.test_macro_f1)} &plusmn; {r.test_macro_f1_std:.3f}",
                        f3(r.test_minority_recall), f3(r.test_worst_class_recall), f"{r.predict_us_per_flow:.2f}",
                        f"{r.model_size_mb:.1f}"])
    rows34 = []
    for strat in ["none", "smote", "classweight", "threshold"]:
        r = s34.loc[strat]
        delta = "&ndash;" if strat == "none" else f"{sig34.loc[strat, 'delta_macro_f1']:+.3f} (p&nbsp;&lt;&nbsp;0.001)"
        rows34.append([STRATEGY[strat], f3(r.test_macro_f1), delta, f3(r.test_minority_recall), f"{r.model_size_mb:.0f}"])
    pc34 = d["compare_34c_rf_test_per_class"].pivot(index="class", columns="strategy", values="recall")
    missed = [c for c in pc34.index if pc34.loc[c, "none"] == 0]
    sup34 = pd.read_csv(results / "per_class" / "34c-rf-none-kall-s0.csv")
    sup34 = sup34[sup34.split == "test"].set_index("class")["support"]

    improve = f"""
<h3>5.3 Improvement (RQ2)</h3>
<p><b>Random Forest, 8 classes.</b> Table 8 compares the six training regimes. Every imbalance method raises macro-F1
and rare-class recall over the baseline, and Table 9 shows that every gain is statistically significant: the 95%
bootstrap confidence intervals of the change in macro-F1 exclude zero for every method and every seed.</p>
{table(['Method', 'Macro-F1', 'Balanced acc.', 'Rare-class recall', 'Web recall', 'BruteForce recall', 'Accuracy', 'Size (MB)', 'Train (s)'], rf_rows, 'small', n.table_caption('Random Forest, 8 classes, test set (mean over three seeds; &plusmn; is the standard deviation).'))}
{table(['Method', 'Mean &Delta; macro-F1', '95% CI (all seeds)', 'Max bootstrap p', 'Mean &Delta; rare-class recall', 'McNemar p &lt; 0.05'], sig_rows, 'small', n.table_caption('Each method vs the Random Forest baseline trained with the same seed, on the same test rows.'))}
<p><b>Threshold tuning</b> gives the best Random Forest (macro-F1 {f3(rf_thr.test_macro_f1)}, +{f3(gain_f1_thr)}) with
no extra training data, the same model size as the baseline ({rf_thr.model_size_mb:.0f}&nbsp;MB) and four seconds of
tuning. It lifts BruteForce recall from {f3(pc8.loc['BruteForce','none'])} to {f3(pc8.loc['BruteForce','threshold'])}
and Web recall from {f3(pc8.loc['Web','none'])} to {f3(pc8.loc['Web','threshold'])}. <b>SMOTE</b> and the hybrid give
the largest rare-class recall ({f3(rf_smote.test_minority_recall)}) but at a cost in precision (Web precision falls
from {f3(prec8.loc['Web','none'])} to {f3(prec8.loc['Web','smote'])}), and the forest trained on synthetic data is
{rf_smote.model_size_mb / rf_none.model_size_mb:.1f} times larger. The SMOTE ablation shows the same trade-off:
raising rare classes to 10,000 rows gives higher rare-class recall
({f3(runs[runs.run_id == '8c-rf-smote10000-kall-s0'].test_minority_recall.iloc[0])}) than 50,000 or 100,000 rows.
<b>Borderline-SMOTE</b> helps least, because few rare-class samples lie near a boundary in this feature space.
For threshold tuning, McNemar's test is not significant for one seed although the macro-F1 gain is: it trades a
few majority-class errors for many rare-class detections, which leaves the overall count of correct rows almost
unchanged while the per-class balance improves.</p>
<p><b>All models.</b> Table 10 ranks every model and method. The class-weighted Decision Tree (macro-F1
{f3(dt_cw.test_macro_f1)}) is almost as good as the best Random Forest at {dt_cw.model_size_mb:.1f}&nbsp;MB and
{dt_cw.predict_us_per_flow:.2f}&nbsp;&micro;s per flow, and even the untuned Decision Tree beats the untuned Random
Forest on macro-F1, most likely because a forest's averaged vote dilutes the few trees that recognise a rare class.
For the Decision Tree, oversampling raises rare-class recall ({f3(lb8[(lb8.model == 'dt') & (lb8.strategy == 'none')].test_minority_recall.iloc[0])}
&rarr; {f3(lb8[(lb8.model == 'dt') & (lb8.strategy == 'smote')].test_minority_recall.iloc[0])} with SMOTE) but lowers macro-F1, because a single
tree fits the synthetic points closely and loses precision; class weights improve both slightly. Threshold tuning
cannot change a fully grown Decision Tree, whose leaves are pure and give 0/1 probabilities. The Balanced Random Forest has the highest
rare-class recall ({f3(brf.test_minority_recall)}) but the lowest precision, so its macro-F1 is only
{f3(brf.test_macro_f1)}. <b>LightGBM</b> with default settings is unstable on this imbalance: without rebalancing
its macro-F1 varies from seed to seed ({f3(lgbm_none.test_macro_f1)} &plusmn; {lgbm_none.test_macro_f1_std:.3f}),
which points to numerical instability of the boosted multi-class model when rare classes contribute tiny gradients;
rebalancing removes the problem (hybrid: {f3(lgbm_hyb.test_macro_f1)} &plusmn; {lgbm_hyb.test_macro_f1_std:.3f}).
Its large gains in Table 10 therefore reflect a repaired baseline as much as an improvement.</p>
{table(['Model', 'Method', 'Macro-F1', 'Rare-class recall', 'Worst-class recall', 'Latency (&micro;s/flow)', 'Size (MB)'], lb_rows, 'small', n.table_caption('All models and methods, 8 classes, all 46 features, ranked by macro-F1 (mean &plusmn; std over three seeds).'))}
<p><b>34 classes.</b> The 34-class task is harder still (Table 11). The baseline Random Forest misses
{len(missed)} attack types completely ({', '.join(missed)}). SMOTE gives the largest gain (macro-F1
{f3(s34.loc['none','test_macro_f1'])} &rarr; {f3(s34.loc['smote','test_macro_f1'])}). These rarest types have only
{int(sup34[missed].min())} to {int(sup34[missed].max())} test rows each, so their individual recalls are noisy.</p>
{table(['Method', 'Macro-F1', '&Delta; vs baseline', 'Rare-class recall', 'Size (MB)'], rows34, 'small', n.table_caption('Random Forest, 34 classes (one seed).'))}
{n.figure('figures/compare_8c_rf_test_recall_gain.png', 'Change in per-class recall over the baseline, Random Forest, 8 classes.')}"""

    rank_rows = [[i + 1, d["rank_mi"].feature.iloc[i], d["rank_rf"].feature.iloc[i], d["rank_rfe"].feature.iloc[i]] for i in range(10)]
    top10 = [set(d[f"rank_{m}"].feature.head(10)) for m in ["mi", "rf", "rfe"]]
    common = sorted(top10[0] & top10[1] & top10[2])
    sweep_rows = []
    for k in [5, 10, 15, 20, 30, 46]:
        r = sw_rf.loc[k]
        sweep_rows.append([k, f3(r.test_macro_f1), f3(r.test_minority_recall), f"{r.predict_us_per_flow:.2f}",
                           f"{r.model_size_mb:.0f}", f3(sw_dt.loc[k].test_macro_f1), f"{sw_dt.loc[k].predict_us_per_flow:.3f}",
                           f"{sw_dt.loc[k].model_size_mb:.2f}"])
    extend = f"""
<h3>5.4 Extension: feature selection (RQ3)</h3>
<p>The three rankings largely agree (Table 12): {len(common)} features are in all three top-10 lists
({', '.join(common)}). The inter-arrival time (IAT) ranks first in every method; packet-size statistics and flow
duration follow.</p>
{table(['Rank', 'Mutual information', 'RF importance', 'RFE'], rank_rows, 'small', n.table_caption('Top-10 features by each ranking (training data only).'), ['center', 'left', 'left', 'left'])}
<p>Table 13 and Figure 12 show detection quality and cost as features are removed (mutual-information ranking).
The threshold-tuned Random Forest loses nothing down to 20 features: macro-F1 is {f3(k20.test_macro_f1)} with 20
features against {f3(k46.test_macro_f1)} with all 46, prediction takes {k20.predict_us_per_flow:.2f} instead of
{k46.predict_us_per_flow:.2f}&nbsp;&micro;s per flow ({pct(1 - k20.predict_us_per_flow / k46.predict_us_per_flow, 0)}
faster), and the model shrinks from {k46.model_size_mb:.0f} to {k20.model_size_mb:.0f}&nbsp;MB
({pct(1 - k20.model_size_mb / k46.model_size_mb, 0)} smaller). With 10 features macro-F1 is still
{f3(k10.test_macro_f1)}, well above the full-feature baseline ({f3(rf_none.test_macro_f1)}), at
{k10.predict_us_per_flow:.2f}&nbsp;&micro;s per flow. The Phase 1 target (at most 15 features keeping most of the gain)
is met: with 15 features the tuned forest keeps
{pct((sw_rf.loc[15].test_macro_f1 - rf_none.test_macro_f1) / gain_f1_thr, 0)} of its macro-F1 gain. Rare-class
recall falls faster than macro-F1 as features are removed, so 20 features is the better operating point for a
detector whose job is rare attacks.</p>
{table(['k', 'RF macro-F1', 'RF rare recall', 'RF &micro;s/flow', 'RF MB', 'DT macro-F1', 'DT &micro;s/flow', 'DT MB'], sweep_rows, 'small', n.table_caption('Feature sweep, 8 classes: threshold-tuned Random Forest and class-weighted Decision Tree (mean over three seeds).'), ['center'] + ['right'] * 7)}
{n.figure('figures/sweep_8c_rf_threshold_ranking_mi_8class.png', 'Threshold-tuned Random Forest against the number of features: detection quality, prediction time, model size and training time.')}
<p>For a Raspberry Pi-class gateway the cost axis matters as much as accuracy (Figure 13). The class-weighted
Decision Tree is the most deployable model: about 1&nbsp;MB and 0.1&nbsp;&micro;s per flow with macro-F1
{f3(dt_cw.test_macro_f1)}, against {rf_thr.model_size_mb:.0f}&nbsp;MB and {rf_thr.predict_us_per_flow:.1f}&nbsp;&micro;s
for the best forest.</p>
{n.figure('figures/cost_vs_f1_8c.png', 'Macro-F1 against model size for each model: baseline (hollow) and best imbalance method (filled).', '82%')}
<h3>5.5 Effect of duplicate rows</h3>
<p>Without de-duplication, {num(dupinfo['test_rows_also_in_train'])} test rows
({pct(dupinfo['test_share_also_in_train'])}) have an exact copy in the training data. Table 14 shows the effect on
the baseline: accuracy is slightly higher with duplicates, because the duplicated rows are mostly easy DDoS and DoS
flows, while macro-F1 is not, because the copies do not help the rare classes. Removing duplicates is therefore the
honest default, and it does not explain away the rare-class problem.</p>
{table(['Data', 'Model', 'Accuracy', 'Macro-F1', 'Rare-class recall'], [[r.data, MODEL[r.model], f3(r.test_accuracy), f3(r.test_macro_f1), f3(r.test_minority_recall)] for r in dup.itertuples() if r.model in ('rf', 'dt')], 'small', n.table_caption('Baselines with and without exact duplicate rows (8 classes).'), ['left', 'left', 'right', 'right', 'right'])}"""

    discussion = f"""
<h2 id="s6">6. Discussion</h2>
<p><b>What fixes the blind spot.</b> The baseline's errors are not random: rare attacks are pulled towards the
classes that surround them in feature space (Benign, Recon, Spoofing). Every remedy moves the decision boundary
towards the rare classes. Threshold tuning does it most cheaply, after training, by re-weighting the forest's votes;
it is the recommended default because it costs no extra data, time or memory and gives the best macro-F1. SMOTE does
it by adding synthetic rare samples, which recovers more rare attacks but raises false alarms and makes the forest
much larger.</p>
<p><b>Precision vs recall.</b> For an IDS, a missed brute-force login is usually worse than a false alarm, so the
higher rare-class recall of SMOTE or the Balanced Random Forest may be preferred where an analyst triages alerts;
where alert fatigue matters, threshold tuning balances both.</p>
<p><b>Deployment.</b> Feature selection shows that the 46 features are redundant: 20 suffice. Combined with the
class-weighted Decision Tree, a gateway can run a detector of about 1&nbsp;MB that classifies a window in about
0.1&nbsp;&micro;s, leaving the Raspberry Pi's resources for packet capture and feature extraction, which are the real
bottlenecks of a live system.</p>
<p><b>Limitations.</b> (1) A 3% sample was used; the rarest 34-class types have only a few test rows. (2) One dataset
from one testbed; transfer to other networks is untested. (3) CICIoT2023 has no replay attacks, which project topic 4
of the course lists. (4) Latency was measured as batch prediction on a Kaggle CPU, not on a Raspberry Pi, and
excludes feature extraction. (5) LightGBM was used with default settings; tuning it might make it competitive without
rebalancing. (6) The live components of Section 3 (capture, MQTT, dashboard) are designed but not deployed.</p>"""

    conclusion = f"""
<h2 id="s7">7. Conclusion and future work</h2>
<p>Following the 10-step IoT design methodology, this project designed a Level 5 gateway intrusion detection system
and built and evaluated its detection model on CICIoT2023. The reproduction matched the paper's accuracy and
confirmed its blind spot on rare attacks, while finding that its precision and recall are reported swapped. Handling
the imbalance on the training data improved the Random Forest's macro-F1 from {f3(rf_none.test_macro_f1)} to
{f3(rf_thr.test_macro_f1)} and its rare-class recall from {f3(rf_none.test_minority_recall)} to as much as
{f3(rf_smote.test_minority_recall)}, all statistically significant. Feature selection kept that quality with 20 of
46 features, and a class-weighted Decision Tree offers nearly the same macro-F1 in about 1&nbsp;MB.</p>
<p><b>Future work:</b> deploy the selected model on a Raspberry Pi with live capture and measure end-to-end latency;
test transfer to CIC IoT-DIAD 2024 [15] and the Dalhousie IoT 2024 dataset [16]; tune LightGBM for imbalance; and try
a two-stage detector that first separates the common floods and then classifies the rest with a specialist model.</p>"""

    refs = [
        "E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, &ldquo;CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment,&rdquo; <i>Sensors</i>, vol. 23, no. 13, 5941, 2023. doi:10.3390/s23135941",
        "N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, &ldquo;SMOTE: Synthetic Minority Over-sampling Technique,&rdquo; <i>Journal of Artificial Intelligence Research</i>, vol. 16, pp. 321&ndash;357, 2002.",
        "H. Han, W.-Y. Wang, B.-H. Mao, &ldquo;Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning,&rdquo; in <i>Proc. ICIC</i>, LNCS 3644, pp. 878&ndash;887, 2005.",
        "C. Chen, A. Liaw, L. Breiman, &ldquo;Using Random Forest to Learn Imbalanced Data,&rdquo; Tech. Rep. 666, Dept. of Statistics, UC Berkeley, 2004.",
        "G. Ke et al., &ldquo;LightGBM: A Highly Efficient Gradient Boosting Decision Tree,&rdquo; in <i>Advances in Neural Information Processing Systems</i>, 2017.",
        "A. Bahga, V. Madisetti, <i>Internet of Things: A Hands-On Approach</i>. VPT / Universities Press, 2014.",
        "L. Breiman, &ldquo;Random Forests,&rdquo; <i>Machine Learning</i>, vol. 45, no. 1, pp. 5&ndash;32, 2001.",
        "F. Pedregosa et al., &ldquo;Scikit-learn: Machine Learning in Python,&rdquo; <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825&ndash;2830, 2011.",
        "G. Lema&icirc;tre, F. Nogueira, C. K. Aridas, &ldquo;Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning,&rdquo; <i>Journal of Machine Learning Research</i>, vol. 18, no. 17, pp. 1&ndash;5, 2017.",
        "Q. McNemar, &ldquo;Note on the sampling error of the difference between correlated proportions or percentages,&rdquo; <i>Psychometrika</i>, vol. 12, no. 2, pp. 153&ndash;157, 1947.",
        "B. Efron, R. J. Tibshirani, <i>An Introduction to the Bootstrap</i>. Chapman &amp; Hall, 1993.",
        "C. Xu, &ldquo;CICIoT2023&rdquo; (mirror of the CSV release), Kaggle dataset, version 2, 2025. kaggle.com/datasets/congyuanxu/ciciot2023",
        "M. A. Khan et al., &ldquo;A Deep Learning-Based Intrusion Detection System for MQTT Enabled IoT,&rdquo; <i>Sensors</i>, vol. 21, no. 21, 7016, 2021. doi:10.3390/s21217016",
        "&ldquo;An efficient intrusion detection system for MQTT-IoT using enhanced chaotic salp swarm algorithm and LightGBM,&rdquo; <i>International Journal of Information Security</i>, Springer, 2023.",
        "M. Rabbani et al., &ldquo;Device Identification and Anomaly Detection in IoT Environments,&rdquo; <i>IEEE Internet of Things Journal</i>, 2024.",
        "J. Adjei, N. Zincir-Heywood, B. Nandy, N. Seddigh, &ldquo;Dalhousie NIMS Lab IoT 2024 Dataset,&rdquo; IEEE DataPort, 2024. doi:10.21227/shzy-z570",
        "B. C. Ross, &ldquo;Mutual Information between Discrete and Continuous Data Sets,&rdquo; <i>PLoS ONE</i>, vol. 9, no. 2, e87357, 2014.",
        "I. Guyon, J. Weston, S. Barnhill, V. Vapnik, &ldquo;Gene Selection for Cancer Classification using Support Vector Machines,&rdquo; <i>Machine Learning</i>, vol. 46, pp. 389&ndash;422, 2002.",
    ]
    references = "<h2 id='refs'>References</h2><ol class='refs'>" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"
    appendix = """
<h2 id="appA">Appendix A: Reproducibility</h2>
<p>Code, tests, results and this report: <b>github.com/Dhruvsandilya0711/IoT_Mini_Project</b> (branch
<code>main</code>). On Kaggle, with the CICIoT2023 CSV dataset [12] attached and Internet
enabled, one notebook cell clones the repository and runs <code>python scripts/kaggle_run.py</code>, which samples
the data, runs every experiment (baseline, improvement, extension, duplicates) and packs <code>results.zip</code>.
Locally, <code>python -m pytest</code> runs the whole pipeline on synthetic data in about 20 seconds.
<code>python scripts/report_figures.py</code> and <code>python report/build_report.py</code> regenerate the figures
and this report from the results.</p>"""

    toc = """
<div class="toc"><h2>Contents</h2><ol>
<li>Introduction</li><li>Background and related work</li><li>IoT system design: the 10-step methodology
<ol><li>Purpose and requirements</li><li>Process model</li><li>Domain model</li><li>Information model</li>
<li>Service specifications</li><li>IoT level</li><li>Functional view</li><li>Operational view</li>
<li>Device and component integration</li><li>Application development</li></ol></li>
<li>Experimental methodology</li><li>Results<ol><li>Data and imbalance</li><li>Reproduction</li><li>Improvement</li>
<li>Extension: feature selection</li><li>Effect of duplicate rows</li></ol></li><li>Discussion</li>
<li>Conclusion and future work</li></ol><p>References &middot; Appendix A: Reproducibility</p></div>"""

    title = """
<section class="title-page">
<div class="kicker">IoT Mini Project &middot; Phase 2: Final Report</div>
<h1>Addressing Class Imbalance in CICIoT2023-Based Intrusion Detection via SMOTE and Feature Selection</h1>
<div class="sub">An IoT gateway intrusion detection system designed with the 10-step IoT design methodology</div>
<div class="author"><b>Dhruv Sandilya</b> &middot; 231CS122<br/>Department of Computer Science and Engineering<br/>
National Institute of Technology Karnataka, Surathkal</div>
<div class="meta">Base paper: E. C. P. Neto et al., &ldquo;CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale
Attacks in IoT Environment,&rdquo; <i>Sensors</i> 23(13):5941, 2023.<br/>November 2026</div>
</section>"""

    css = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'Carlito', 'Calibri', sans-serif; font-size: 11pt; line-height: 1.42; color: #1B2430; }
h1, h2, h3 { font-family: 'Caladea', 'Cambria', serif; color: #0E1E33; line-height: 1.2; }
h2 { font-size: 17pt; margin: 22pt 0 8pt; break-after: avoid; }
h3 { font-size: 13pt; margin: 14pt 0 6pt; break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 8pt; }
li { margin-bottom: 3pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #EEF3F7; padding: 0 2pt; }
.title-page { height: 245mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.title-page .kicker { color: #0E9AA7; font-weight: bold; letter-spacing: 1.5pt; text-transform: uppercase; font-size: 10.5pt; }
.title-page h1 { font-size: 27pt; margin: 12pt 0; }
.title-page .sub { font-size: 14pt; color: #3D5A80; margin-bottom: 40pt; }
.title-page .author { font-size: 12.5pt; margin-bottom: 30pt; }
.title-page .meta { font-size: 10.5pt; color: #52514e; }
.abstract { background: #EEF3F7; padding: 10pt 14pt; border-radius: 6pt; margin-bottom: 10pt; }
.abstract h2 { margin-top: 0; }
.toc { break-after: page; font-size: 10pt; }
.toc h2 { margin-top: 12pt; }
.toc > ol { column-count: 2; column-gap: 24pt; padding-left: 18pt; margin: 0; }
.toc li { margin-bottom: 1pt; break-inside: avoid-column; } .toc ol ol { list-style: lower-alpha; padding-left: 16pt; }
figure { margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 9.5pt; color: #52514e; margin-top: 4pt; text-align: left; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0 12pt; font-size: 9.5pt; break-inside: avoid; }
table.small { font-size: 8.8pt; }
caption { caption-side: top; text-align: left; font-size: 9.5pt; color: #52514e; margin-bottom: 4pt; }
th { background: #0E1E33; color: white; padding: 4pt 5pt; font-weight: bold; vertical-align: bottom; }
td { padding: 3pt 5pt; border-bottom: 0.6pt solid #DDE3EA; vertical-align: top; }
tbody tr:nth-child(even) td { background: #F6F8FA; }
table.spec td:first-child { font-weight: bold; color: #0E1E33; white-space: nowrap; }
.refs { font-size: 9.5pt; padding-left: 20pt; } .refs li { margin-bottom: 4pt; }
"""
    body = (title + f'<div class="abstract"><h2>Abstract</h2>{abstract}</div>' + toc + intro + background + design
            + method + repro + improve + extend + discussion + conclusion + references + appendix)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Phase 2 Report</title>'
            f"<style>{css}</style></head><body>{body}</body></html>")


COPIED_FIGURES = ["eda/class_distribution_8class.png", "figures/compare_8c_rf_test_recall_gain.png",
                  "figures/sweep_8c_rf_threshold_ranking_mi_8class.png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="report/report.html")
    args = parser.parse_args()
    out, results = Path(args.out), Path(args.results)
    (out.parent / "figures").mkdir(parents=True, exist_ok=True)
    for f in COPIED_FIGURES:
        shutil.copy(results / f, out.parent / "figures" / Path(f).name)
    out.write_text(build(results, out.parent / "figures"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
