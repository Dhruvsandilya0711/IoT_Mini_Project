"""Build the mini project report (HTML, then PDF via render_pdf.cjs) from the result files.

    python report/build_report.py --results results --out report/report.html
    node report/render_pdf.cjs report/report.html report/Mini_Project_Report.pdf

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

TITLE = "Addressing Class Imbalance in CICIoT2023-Based Intrusion Detection via SMOTE and Feature Selection"
AUTHOR, ROLL = "Dhruv Sandilya", "231CS122"
DATE = "November 2026"


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


def answer(label, text):
    return f'<div class="answer"><span class="tag">{label}</span><p>{text}</p></div>'


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


class Headings:
    """Numbered headings; remembers them so the contents page can list them (render_pdf.cjs adds page numbers)."""

    def __init__(self):
        self.entries = []

    def h2(self, number, title, anchor):
        self.entries.append((2, number, title))
        lead = f'<span class="num">{number}</span> ' if number else ""
        return f'<h2 class="sec" id="{anchor}">{lead}{title}</h2>'

    def h3(self, number, title):
        self.entries.append((3, number, title))
        return f'<h3><span class="num">{number}</span> {title}</h3>'

    def contents(self):
        items = []
        for level, number, title in self.entries:
            find = f"{number} {title}".strip()
            items.append(f'<li class="l{level - 1}" data-find="{find}"><span class="n">{number}</span>'
                         f'<span class="t">{title}</span><span class="dots"></span><span class="pg"></span></li>')
        return f'<section class="toc"><h2 class="plain">Contents</h2><ol>{"".join(items)}</ol></section>'


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
    H = Headings()
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
    lb8 = d["leaderboard_8c"]
    sw_rf = d["sweep_8c_rf_threshold_ranking_mi_8class"].set_index("n_features")
    sw_dt = d["sweep_8c_dt_classweight_ranking_mi_8class"].set_index("n_features")
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
    faster20 = pct(1 - k20.predict_us_per_flow / k46.predict_us_per_flow, 0)
    smaller20 = pct(1 - k20.model_size_mb / k46.model_size_mb, 0)

    def sig_summary(sig, strategy):
        s = sig[sig.strategy == strategy]
        return (s.delta_macro_f1.mean(), s.delta_macro_f1_low.min(), s.delta_macro_f1_high.max(),
                s.delta_minority_recall.mean(), int((s.p_mcnemar < 0.05).sum()), len(s), s.p_macro_f1.max())

    # ---------- front matter ----------
    cover = f"""
<section class="cover">
<img class="logo" src="figures/nitk_logo.png" alt="NITK emblem"/>
<div class="inst">National Institute of Technology Karnataka<br/>Surathkal</div>
<div class="dept">Department of Computer Science and Engineering</div>
<div class="rule"></div>
<div class="kicker">Internet of Things &middot; Mini Project Report</div>
<h1>{TITLE}</h1>
<div class="spacer"></div>
<div class="by"><div class="label">Submitted by</div><div class="name">{AUTHOR}</div><div class="roll">{ROLL}</div></div>
<div class="base"><b>Base paper:</b> E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani,
&ldquo;CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment,&rdquo;
<i>Sensors</i>, vol. 23, no. 13, 5941, 2023.</div>
<div class="date">{DATE}</div>
</section>"""

    abstract = f"""
<section class="front">
<h2 class="plain">Abstract</h2>
<p>CICIoT2023, published by Neto et al., is one of the largest public datasets for intrusion detection in IoT
networks: {num(total_rows)} labelled flows recorded on a testbed of 105 devices under 33 attacks. The paper's Random
Forest scores above 99% accuracy, but that number hides a problem. DDoS and DoS make up
{pct(counts8.loc['DDoS','share'] + counts8.loc['DoS','share'])} of the data, and there are
{num(counts8.loc['DDoS','count'] / counts8.loc['BruteForce','count'])} DDoS flows for every brute-force flow, so the
rare attacks are mostly missed.</p>
<p>In this project I design a gateway intrusion detection system with the 10-step IoT design methodology and build its
detection model in three stages. First, I reproduce the paper's Random Forest on a de-duplicated
{pct(d['sample']['frac'])} sample ({num(d['sample']['rows'])} flows drawn from all 169 files). It matches the paper's
accuracy ({f3(rf_none.test_accuracy)} against {f3(PAPER_RF[8]['accuracy'])} for 8 classes) and has the same weakness:
web attacks are recognised only {pct(pc8.loc['Web','none'],0)} of the time and brute-force attacks
{pct(pc8.loc['BruteForce','none'],0)}. Second, I try five ways of handling the imbalance on the training data (SMOTE,
Borderline-SMOTE, SMOTE with undersampling, class weights and decision-threshold tuning) with four models. For the
Random Forest all five give a statistically significant gain (paired bootstrap, p&nbsp;&lt;&nbsp;0.05). Threshold
tuning works best, raising macro-F1 from {f3(rf_none.test_macro_f1)} to {f3(rf_thr.test_macro_f1)} without making the
model any bigger, while SMOTE gives the highest rare-class recall ({f3(rf_none.test_minority_recall)} to
{f3(rf_smote.test_minority_recall)}). Third, I use feature selection to make the detector lighter. With 20 of the 46
features the tuned forest keeps its macro-F1 ({f3(k20.test_macro_f1)}) while predicting {faster20} faster from a
{smaller20} smaller model, and a class-weighted Decision Tree reaches macro-F1 {f3(dt_cw.test_macro_f1)} in only
{dt_cw.model_size_mb:.1f}&nbsp;MB.</p>
<p>Two side findings are worth reporting. Unless duplicate rows are removed,
{pct(dupinfo['test_share_also_in_train'])} of the test rows also appear in the training data, and the precision and
recall columns in the paper's results table are swapped.</p>
<div class="glance-label">Results at a glance</div>
<div class="glance">
<div><b>{f3(rf_none.test_accuracy)}</b><span>accuracy of the reproduced Random Forest, 8 classes (paper: {f3(PAPER_RF[8]['accuracy'])})</span></div>
<div><b>+{f3(gain_f1_thr)}</b><span>macro-F1 from threshold tuning ({f3(rf_none.test_macro_f1)} &rarr; {f3(rf_thr.test_macro_f1)})</span></div>
<div><b>{pct(pc8.loc['BruteForce','none'],0)} &rarr; {pct(pc8.loc['BruteForce','threshold'],0)}</b><span>brute-force recall, baseline to threshold tuning (paper: {pct(paper['per_class_recall']['BruteForce'],0)})</span></div>
<div><b>20 / 46</b><span>features keep the macro-F1, {faster20} faster and {smaller20} smaller</span></div>
</div>
<p class="keywords"><b>Keywords:</b> IoT security, intrusion detection, CICIoT2023, class imbalance, SMOTE, threshold
tuning, feature selection, edge computing</p>
</section>"""

    # ---------- 1. Introduction ----------
    intro = f"""
{H.h2('1', 'Introduction', 's1')}
{H.h3('1.1', 'Motivation')}
<p>Most IoT devices are small, cheap and rarely updated, and every one that joins a home or factory network gives an
attacker another way in. Intrusion detection systems (IDS) that learn from labelled traffic are a common defence, and
CICIoT2023 [1] was created to train and test such systems on realistic IoT traffic. The trouble is its class
distribution. A detector that always answered &ldquo;DDoS&rdquo; would already be
{pct(counts8.loc['DDoS','share'],0)} accurate, yet it would never catch a brute-force login or a web injection, and
those are exactly the attacks that let someone take over a device. Accuracy on its own therefore says very little
about the attacks we care about most, so this report looks at the recall of each class instead.</p>
{H.h3('1.2', 'Base paper')}
<p>Neto et al. [1] built a smart-home testbed of 105 devices (67 IP devices and 38 Zigbee and Z-Wave devices behind
five hubs), launched 33 attacks from compromised IoT devices and released {num(total_rows)} labelled rows. Each row
has 46 features computed over a window of 10 or 100 packets. They trained Logistic Regression, a Perceptron,
AdaBoost, a Random Forest and a deep neural network on 2-, 8- and 34-class versions of the problem, using an 80/20
split and standard scaling, and did nothing special about the class imbalance. Their own confusion matrix (Table 8 of
[1]) shows the Random Forest recognising only {pct(paper['per_class_recall']['Web'])} of web-based flows and
{pct(paper['per_class_recall']['BruteForce'])} of brute-force flows. In their conclusion they list &ldquo;the
analysis of features and how they influence different ML models&rdquo; as future work, and that is where the
extension in this project comes from.</p>
{H.h3('1.3', 'Objectives and contributions')}
<p>The project is built around three research questions, each with a target that was fixed in Phase 1.</p>
<ul class="rq">
<li><b>RQ1, reproduction.</b> Can the paper's Random Forest benchmark be reproduced, and how do a Decision Tree,
LightGBM and a Balanced Random Forest compare? <i>Yes. Accuracy agrees with the paper to within 0.002 on all three
tasks (Section 5.2).</i></li>
<li><b>RQ2, improvement.</b> Does handling the imbalance raise recall on the rare classes, and is the gain
significant? <i>Yes. For the Random Forest every method beats the baseline (p&nbsp;&lt;&nbsp;0.05). Rare-class
recall rises by up to {f3(gain_min_smote)} and macro-F1 by {f3(gain_f1_thr)} (Section 5.3).</i></li>
<li><b>RQ3, extension.</b> Can a smaller set of features keep that gain at a lower cost? <i>Yes. 20 of the 46
features keep the full macro-F1, with faster prediction and a smaller model (Section 5.4).</i></li>
</ul>
<p>A few things came up that I had not planned for. Unless duplicate rows are removed,
{pct(dupinfo['test_share_also_in_train'])} of the test set also appears in the training data. The precision and
recall columns in the paper's Table 6 are swapped. And LightGBM with default settings turns out to be unstable on
this data unless the training set is rebalanced.</p>
{H.h3('1.4', 'Organisation of the report')}
<p>Section 2 gives some background. Section 3 designs the system with Bahga and Madisetti's 10-step IoT design
methodology [6]. Its last step, application development, is the machine-learning pipeline, and Sections 4 and 5
describe how that pipeline was evaluated and what it found. Section 6 discusses the results and Section 7
concludes.</p>"""

    # ---------- 2. Background ----------
    background = f"""
{H.h2('2', 'Background and related work', 's2')}
<p><b>Datasets.</b> What sets CICIoT2023 [1] apart is its size and the fact that the attacks are launched from IoT
devices themselves. The course reading list also includes CIC IoT-DIAD 2024 [15], built for device identification
and anomaly detection, and the Dalhousie NIMS Lab IoT 2024 dataset [16]. Both would be good places to test whether
models trained here carry over to other networks.</p>
<p><b>IoT intrusion detection.</b> Deep-learning IDS have been proposed for MQTT-based IoT networks [13], and LightGBM
has been combined with metaheuristic feature selection for MQTT intrusion detection [14]. On tabular flow features,
though, tree ensembles such as the Random Forest [7] are still strong and inexpensive baselines, which is why this
project builds on them.</p>
<p><b>Class imbalance.</b> There are two broad families of fixes. Data-level methods change what the model is
trained on: SMOTE [2] creates synthetic minority samples by interpolating between neighbouring ones, Borderline-SMOTE
[3] does this only close to the class boundary, and random undersampling throws away some of the majority samples.
Algorithm-level methods leave the data alone: class weights make mistakes on rare classes cost more, threshold moving
changes the decision rule after training, and the Balanced Random Forest [4] trains each tree on a class-balanced
bootstrap sample. I also include LightGBM [5] as a strong, modern gradient-boosting model.</p>
<p><b>Feature selection.</b> Mutual information between a feature and the label [17], the impurity importance of a
Random Forest [7] and recursive feature elimination [18] are standard examples of filter, embedded and wrapper
methods respectively.</p>
<p><b>IoT design methodology.</b> Bahga and Madisetti [6] describe ten steps that take an IoT system from its purpose
and requirements to application development, and define six deployment levels. Section 3 works through them for this
system.</p>"""

    # ---------- 3. IoT system design ----------
    req_rows = [
        ["Purpose", "Detect and classify malicious traffic in an IoT network at its gateway, including rare attack types, so that an administrator can respond."],
        ["Behaviour", "Watch the traffic of every connected device, summarise it into packet windows, label each window as benign or as one of 7 attack categories (33 attack types), and raise alerts."],
        ["System management", "Monitor the gateway's health and throughput remotely, and update the detection model remotely."],
        ["Data analysis", "Real-time inference at the gateway (edge); training, rebalancing and feature selection in the cloud."],
        ["Application deployment", "Detection on the gateway; storage, dashboard and model registry in the cloud."],
        ["Security", "Encrypted, authenticated channels (TLS) for alerts and updates; an integrity check on every model before it is installed; the IDS only sees a copy of the traffic, so the network keeps running if it fails."],
        ["Measurable targets", "Rare-class recall and macro-F1 above the reproduced baseline (significant at p &lt; 0.05); prediction time in microseconds per flow; a model small enough for a Raspberry Pi-class gateway."],
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
        ["3", "Single node; cloud storage and analysis", "No: single node, and sending raw traffic to the cloud costs too much"],
        ["4", "Multiple nodes, each analysing locally; cloud storage", "No: IoT devices are too constrained to run an IDS each"],
        ["<b>5</b>", "<b>Multiple end nodes and one coordinator; cloud storage and application</b>", "<b>Yes: the devices are end nodes and the gateway is the coordinator</b>"],
        ["6", "Independent end nodes sending to the cloud; central controller", "No: devices cannot run agents, and sending all traffic to the cloud adds latency"],
    ]
    op_rows = [
        ["Device", "Raspberry Pi 4 gateway with a network tap or switch mirror port; cloud VM"],
        ["Communication", "MQTT broker (e.g. Mosquitto) with TLS for alerts; HTTPS for model updates and the dashboard"],
        ["Services hosting", "Capture, feature extraction and detection run natively on the gateway; storage, dashboard and registry on the cloud VM"],
        ["Storage", "Relational or time-series database for alerts; object storage for model artifacts"],
        ["Training", "Batch training on a Kaggle CPU (used in this project; 2.7 h for the full experiment matrix)"],
        ["Application hosting", "Web dashboard on the cloud VM"],
        ["Management", "Versioned model artifacts with metrics and a SHA-256 checksum; a gateway configuration file (top-k feature list, class weights)"],
        ["Security", "X.509 client certificates for MQTT; TLS everywhere; checksum verified before a model is installed"],
    ]
    mod_rows = [
        ["src/load.py", "Streams the 169 CSV files and keeps a random 3% of the rows (class ratios preserved)"],
        ["src/preprocess.py", "Removes NaN/inf rows and exact duplicates; makes the frozen, stratified train/val/test split"],
        ["src/resample.py, src/threshold.py", "Imbalance methods (training data only); per-class threshold tuning on the validation set"],
        ["src/train.py", "Trains and evaluates one configuration; logs metrics, latency, size and predictions"],
        ["src/feature_selection.py", "Mutual-information, RF-importance and RFE rankings on the training data"],
        ["src/stats.py, src/analyze.py", "Bootstrap confidence intervals, McNemar tests, tables and figures"],
        ["scripts/kaggle_run.py", "Runs every phase on Kaggle and packs the results"],
        ["scripts/pi_detect.py", "Replays recorded flows through a saved model on a Raspberry Pi gateway"],
    ]
    design = f"""
{H.h2('3', 'IoT system design', 's3')}
<p>The system is an intrusion detection system that sits at the gateway of an IoT network. This section designs it
step by step with the 10-step methodology of Bahga and Madisetti [6]; Figure 1 gives an overview of how each step
applies.</p>
{n.figure('figures/iot_methodology_overview.png', 'The ten steps of the IoT design methodology applied to the gateway intrusion detection system.')}
{H.h3('3.1', 'Step 1: Purpose and requirements specification')}
{table(['Aspect', 'Specification'], req_rows, 'spec', n.table_caption('Purpose and requirements.'), ['left', 'left'])}
{H.h3('3.2', 'Step 2: Process model specification')}
<p>The process model (Figure 2) is a flowchart of how the system works. It has two loops. The online loop runs on
the gateway: it classifies every window of traffic and raises alerts. The offline loop runs in the cloud: it trains
and evaluates new models and publishes the good ones. The two loops meet at the model update service and the alert
store.</p>
{n.figure('figures/iot_process_model.png', 'Process model: online detection at the gateway (A) and offline training in the cloud (B).', '86%')}
{H.h3('3.3', 'Step 3: Domain model specification')}
<p>The domain model (Figure 3) lists the entities, objects and relationships without tying them to any technology.
The <b>physical entities</b> are the IoT devices, the malicious devices and the traffic between them. Each has a
<b>virtual entity</b> standing in for it: a device profile, the flow windows of its traffic, detection results,
alerts and the model itself. The <b>devices</b> of the system, namely the gateway with its network tap and a cloud
server, observe the physical entities and host the <b>resources</b>. Some resources live on the device (packet
capture, feature extractor, classifier, local log) and others on the network (alert store, model registry, training
compute). <b>Services</b> make these resources available to the administrator.</p>
{n.figure('figures/iot_domain_model.png', 'Domain model: physical and virtual entities and the devices, resources and services that link them.')}
{H.h3('3.4', 'Step 4: Information model specification')}
<p>The information model (Figure 4) defines the data structures behind the virtual entities and how they connect. The
central one is the <b>FlowWindow</b>, which holds the 46 CICIoT2023 features for one window of 10 or 100 packets
between two hosts [1]. A Detection refers to one FlowWindow and to the ModelArtifact that produced it, and an Alert
groups the detections for one device.</p>
{n.figure('figures/iot_information_model.png', 'Information model: virtual entities, their attributes and their relationships.', '92%')}
{H.h3('3.5', 'Step 5: Service specifications')}
<p>Table 2 lists the services, what each one takes in and produces, how often it runs and where it is hosted.</p>
{table(['Service', 'Type', 'Input', 'Output', 'Schedule', 'Hosted on'], svc_rows, 'spec small', n.table_caption('Service specifications.'), ['left'] * 6)}
{H.h3('3.6', 'Step 6: IoT level specification')}
<p>The system is a <b>Level 5</b> deployment (Figure 5). The IoT devices are end nodes that only generate traffic,
the gateway is the coordinator that sees all of it, and the cloud handles storage, analysis and the application.
There is one intentional difference from the textbook Level 5: detection runs on the coordinator instead of in the
cloud, so an attack can be flagged within microseconds and the raw traffic never has to leave the network. Keeping
that edge inference cheap is the reason for the feature-selection extension in Section 5.4. Table 3 explains why the
other levels were ruled out.</p>
{n.figure('figures/iot_iot_level.png', 'Level 5 deployment: IoT devices as end nodes, the gateway as coordinator with edge detection, and the cloud.', '92%')}
{table(['Level', 'Definition [6]', 'Fit for this system'], level_rows, 'spec', n.table_caption('Choosing the IoT level.'), ['center', 'left', 'left'])}
{H.h3('3.7', 'Step 7: Functional view specification')}
<p>The functional view (Figure 6) maps the entities and services onto the IoT-A functional groups: Device,
Communication, Services, Management, Security and Application.</p>
{n.figure('figures/iot_functional_view.png', 'Functional view: the components of each functional group.')}
{H.h3('3.8', 'Step 8: Operational view specification')}
<p>Table 4 records the concrete choices for deploying and running each functional group.</p>
{table(['Functional group', 'Operational choice'], op_rows, 'spec', n.table_caption('Operational view: how each functional group is deployed and operated.'), ['left', 'left'])}
{H.h3('3.9', 'Step 9: Device and component integration')}
<p>Figure 7 shows how the components connect. The IoT devices join an access point or switch as usual. A network
tap, as in the CICIoT2023 testbed [1], or a switch mirror port gives the gateway a passive copy of all traffic, so if
the IDS fails the network keeps running. The gateway sends alerts to the cloud and pulls model updates from it.</p>
{n.figure('figures/iot_integration.png', 'Device and component integration.')}
{H.h3('3.10', 'Step 10: Application development')}
<p>At the heart of the application is the detection model and the pipeline that trains and evaluates it. I wrote it
as a tested Python package using scikit-learn [8], imbalanced-learn [9] and LightGBM [5] and ran it end to end on
Kaggle; Table 5 lists its modules. Sections 4 and 5 describe the experiments used to choose the model that goes on
the gateway.</p>
{table(['Module', 'Role'], mod_rows, 'spec', n.table_caption('Software modules of the application (repository: github.com/Dhruvsandilya0711/IoT_Mini_Project).'), ['left', 'left'])}"""

    # ---------- 4. Methodology ----------
    method = f"""
{H.h2('4', 'Experimental methodology', 's4')}
{H.h3('4.1', 'Data')}
<p>I streamed all 169 CSV files of the original release and kept each row with probability
{d['sample']['frac']:.4f}. This gave {num(d['sample']['rows'])} rows ({pct(d['sample']['rows'] / total_rows, 2)} of
{num(total_rows)}) with the original class ratios intact. No row had missing or infinite values. I removed
{num(clean['duplicates_dropped'])} exact duplicate rows ({pct(clean['duplicates_dropped'] / clean['rows_in'])}) and
split the remaining {num(clean['rows_out'])} rows once, stratified on all 34 classes, into training
({num(rows_split['train'])}), validation ({num(rows_split['val'])}) and test ({num(rows_split['test'])}) sets. The
test set is never resampled and is never used to make any choice. I call the classes that make up less than 1% of
the training data (Recon, Web and BruteForce) the <i>rare classes</i>, and report their mean recall as rare-class
recall.</p>
{H.h3('4.2', 'Models')}
<p>I used four models: a Decision Tree; a Random Forest with 100 trees, as in [1]; LightGBM with 100 boosting rounds,
63 leaves and a learning rate of 0.1; and a Balanced Random Forest with 100 trees, each trained on a class-balanced
bootstrap sample. Everything else is left at the library defaults, and the features are standardised as in [1].</p>
{H.h3('4.3', 'Imbalance methods')}
<p>All of these are applied to the training data only.</p>
<ul>
<li><b>SMOTE:</b> every class with fewer than 50,000 training rows is topped up to 50,000 with synthetic samples
(k&nbsp;=&nbsp;5). An ablation also tries 10,000 and 100,000.</li>
<li><b>SMOTE + undersampling (hybrid):</b> classes with more than 200,000 rows are first cut down to 200,000 at
random, and then SMOTE is applied as above.</li>
<li><b>Borderline-SMOTE:</b> like SMOTE, but only minority samples near the class boundary are used.</li>
<li><b>Class weights:</b> each class's share of the loss is weighted by the inverse of its frequency.</li>
<li><b>Threshold tuning:</b> the model is trained on the original data. Afterwards each class's predicted
probability is multiplied by a weight, and the weights are chosen by coordinate ascent over 22 values between 0.1 and
100 to maximise macro-F1 on the <i>validation</i> set.</li>
</ul>
{H.h3('4.4', 'Feature selection')}
<p>Features are ranked on the training data only, in three ways: mutual information [17] and Random Forest importance
(both on a stratified subsample of 200,000 rows), and RFE [18] with a 50-tree forest (on 50,000 rows). Each model is
then retrained on the top k&nbsp;&isin;&nbsp;{{5, 10, 15, 20, 30}} features of the mutual-information ranking, using
whichever imbalance method gave it the best macro-F1 on the <i>validation</i> set.</p>
{H.h3('4.5', 'Metrics and statistics')}
<p>On the test set I report accuracy, macro-averaged precision, recall and F1, balanced accuracy, the Matthews
correlation coefficient, the recall of each class and rare-class recall. For cost, I measure prediction latency
(batch prediction time per flow on a 4-core Kaggle CPU) and model size (the size of the saved model). Every 8-class
configuration is run with three seeds and reported as mean &plusmn; standard deviation. To check whether a method
really beats the baseline, I compare it with the baseline trained with the same seed on the same test rows, using a
paired bootstrap (1,000 resamples, giving a 95% confidence interval and a one-sided p-value) [11] and McNemar's exact
test on which rows each model gets right [10].</p>
{H.h3('4.6', 'Execution')}
<p>The whole experiment matrix, {len(runs)} training runs, ran unattended on a free Kaggle CPU session and took 2.7
hours.</p>"""

    # ---------- 5. Results ----------
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
{H.h2('5', 'Results', 's5')}
{H.h3('5.1', 'Data and imbalance')}
<p>Table 6 and Figure 8 confirm the imbalance described in [1]. The category shares in my sample match the paper's
Table 3 to within sampling error, and the rarest category, BruteForce, is
{num(counts8.loc['BruteForce','imbalance_ratio'])} times smaller than DDoS. Only {num(per_cat['Web']['test'])} Web and
{num(per_cat['BruteForce']['test'])} BruteForce flows end up in the test set, which is why their recall has the
widest confidence intervals. Two features, Telnet and SMTP, are zero in every row of the sample.</p>
{table(['Category', 'Rows in sample', 'Share', 'DDoS : class', 'Train rows', 'Test rows'], data_rows, '', n.table_caption('Class distribution of the sample (counts and shares before de-duplication; split sizes after it).'))}
{n.figure('figures/class_distribution_8class.png', 'Class distribution of the 8-class task (log scale).', '80%')}
{H.h3('5.2', 'Reproduction (RQ1)')}
<p>Table 7 puts the reproduced Random Forest next to the paper's. Accuracy agrees to within 0.002 for every task.
Precision and recall need more care. The 34-class Logistic Regression row of the paper's Table 6 (accuracy
0.80231507, &ldquo;recall&rdquo; 0.59520185, &ldquo;precision&rdquo; 0.486752461, F1 0.49388408) matches, digit for
digit, the output of the public CICIoT2023 example notebook. That notebook calls <code>recall_score(y_pred,
y_test)</code> and <code>precision_score(y_pred, y_test)</code>, but scikit-learn expects <code>(y_true,
y_pred)</code>, so the two metrics come out the wrong way round. The paper's own Random Forest confusion matrix (its
Table 8) confirms this. Recomputing from it gives a macro recall of {f3(paper['recall'])} and a macro precision of
{f3(paper['precision'])}, the reverse of the values printed in Table 6 ({f3(PAPER_RF[8]['recall'])} and
{f3(PAPER_RF[8]['precision'])}), while accuracy ({f3(paper['accuracy'])}) and F1 ({f3(paper['f1'])}) agree. In
Table 7 I therefore compare the paper's &ldquo;recall&rdquo; with my precision, and the other way round.</p>
{table(['Task', 'Accuracy, paper', 'Accuracy, ours', 'Macro-F1, paper', 'Macro-F1, ours', 'Paper: printed recall / precision', 'Ours: precision / recall'], paper_rows, 'small', n.table_caption('Random Forest: base paper (Table 6 of [1]) against this reproduction. Ours are means over three seeds for 8 classes and one seed otherwise.'))}
<p>My macro-F1 is higher than the paper's for 8 and 34 classes. The split is different (random, stratified and
de-duplicated), and library versions and training details probably differ too. One detail stands out: the same
example notebook calls <code>fit()</code> once for each CSV file, and in scikit-learn every call retrains the model
from scratch, so a model trained that way only learns from the last file, about 0.6% of the training data. What
matters more is that the overall pattern is the same (Figure 9). DDoS, DoS and Mirai are recognised almost perfectly,
while Web and BruteForce flows are mostly mistaken for Benign, Recon or Spoofing, just as in the paper's Table 8
(Figure 10).</p>
{n.figure('figures/repro_per_class_8c.png', "Per-class recall of the Random Forest, 8 classes: the base paper (from its Table 8) and this project's baseline, SMOTE and threshold-tuned models (mean over three seeds).")}
{n.figure('figures/confusion_rf_pair.png', 'Random Forest confusion matrices (8 classes, seed 0), row-normalised: baseline against threshold tuning.')}
{answer('Answer to RQ1', f"Yes. The reproduced Random Forest matches the paper's accuracy to within 0.002 on all three tasks and has the same weakness: Web and BruteForce flows are recalled only {pct(pc8.loc['Web','none'],0)} and {pct(pc8.loc['BruteForce','none'],0)} of the time.")}"""

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
    dt_none_rare = lb8[(lb8.model == 'dt') & (lb8.strategy == 'none')].test_minority_recall.iloc[0]
    dt_smote_rare = lb8[(lb8.model == 'dt') & (lb8.strategy == 'smote')].test_minority_recall.iloc[0]
    smote10k_rare = runs[runs.run_id == '8c-rf-smote10000-kall-s0'].test_minority_recall.iloc[0]

    improve = f"""
{H.h3('5.3', 'Improvement (RQ2)')}
<p><b>Random Forest, 8 classes.</b> Table 8 compares the six ways of training. Every imbalance method improves both
macro-F1 and rare-class recall over the baseline, and Table 9 shows that the gains are real: for every method and
every seed, the 95% bootstrap confidence interval for the change in macro-F1 lies above zero.</p>
{table(['Method', 'Macro-F1', 'Balanced acc.', 'Rare-class recall', 'Web recall', 'BruteForce recall', 'Accuracy', 'Size (MB)', 'Train (s)'], rf_rows, 'small', n.table_caption('Random Forest, 8 classes, test set (mean over three seeds; &plusmn; is the standard deviation).'))}
{table(['Method', 'Mean &Delta; macro-F1', '95% CI (all seeds)', 'Max bootstrap p', 'Mean &Delta; rare-class recall', 'McNemar p &lt; 0.05'], sig_rows, 'small', n.table_caption('Each method against the Random Forest baseline trained with the same seed, on the same test rows.'))}
<p><b>Threshold tuning</b> gives the best Random Forest, with a macro-F1 of {f3(rf_thr.test_macro_f1)}
(+{f3(gain_f1_thr)}). It needs no extra training data, adds about four seconds of tuning and leaves the model the same
size as the baseline ({rf_thr.model_size_mb:.0f}&nbsp;MB). It raises BruteForce recall from
{f3(pc8.loc['BruteForce','none'])} to {f3(pc8.loc['BruteForce','threshold'])} and Web recall from
{f3(pc8.loc['Web','none'])} to {f3(pc8.loc['Web','threshold'])}. <b>SMOTE</b> and the hybrid reach the highest
rare-class recall ({f3(rf_smote.test_minority_recall)}), but they pay for it in precision: Web precision drops from
{f3(prec8.loc['Web','none'])} to {f3(prec8.loc['Web','smote'])}, and the forest trained on synthetic data is
{rf_smote.model_size_mb / rf_none.model_size_mb:.1f} times larger. The SMOTE ablation shows the same trade-off, since
topping the rare classes up to only 10,000 rows gives a higher rare-class recall ({f3(smote10k_rare)}) than 50,000 or
100,000 rows. <b>Borderline-SMOTE</b> helps least, probably because few rare-class samples lie near a class boundary
in this feature space.</p>
<p>One result looks odd at first. For one seed, McNemar's test is not significant for threshold tuning even though
the macro-F1 gain is. The reason is that threshold tuning swaps a few errors on the common classes for many correct
rare-class detections. The total number of correctly labelled rows hardly changes, but the balance across classes
improves a lot.</p>
<p><b>All models.</b> Table 10 ranks every model and method. The class-weighted Decision Tree (macro-F1
{f3(dt_cw.test_macro_f1)}) comes close to the best Random Forest while taking only {dt_cw.model_size_mb:.1f}&nbsp;MB
and {dt_cw.predict_us_per_flow:.2f}&nbsp;&micro;s per flow. Even the plain Decision Tree beats the plain Random
Forest on macro-F1, most likely because the forest averages its trees' votes and drowns out the few trees that
recognise a rare class. For the Decision Tree, oversampling raises rare-class recall ({f3(dt_none_rare)} &rarr;
{f3(dt_smote_rare)} with SMOTE) but lowers macro-F1, because a single tree fits the synthetic points too closely and
loses precision; class weights improve both a little. Threshold tuning cannot change a fully grown Decision Tree at
all, since its leaves are pure and its probabilities are always 0 or 1. The Balanced Random Forest has the highest
rare-class recall ({f3(brf.test_minority_recall)}) but the lowest precision, so its macro-F1 is only
{f3(brf.test_macro_f1)}.</p>
<p><b>LightGBM</b> with default settings struggles with this imbalance. Without rebalancing its macro-F1 varies a lot
from seed to seed ({f3(lgbm_none.test_macro_f1)} &plusmn; {lgbm_none.test_macro_f1_std:.3f}), which suggests
numerical instability when the rare classes contribute only tiny gradients. Rebalancing fixes this (hybrid:
{f3(lgbm_hyb.test_macro_f1)} &plusmn; {lgbm_hyb.test_macro_f1_std:.3f}), so its large gains in Table 10 are as much a
repaired baseline as a real improvement.</p>
{table(['Model', 'Method', 'Macro-F1', 'Rare-class recall', 'Worst-class recall', 'Latency (&micro;s/flow)', 'Size (MB)'], lb_rows, 'small', n.table_caption('All models and methods, 8 classes, all 46 features, ranked by macro-F1 (mean &plusmn; standard deviation over three seeds).'))}
<p><b>34 classes.</b> With 34 classes the task is harder again (Table 11). The baseline Random Forest never correctly
identifies {len(missed)} attack types ({', '.join(missed)}). SMOTE gives the biggest gain, raising macro-F1 from
{f3(s34.loc['none','test_macro_f1'])} to {f3(s34.loc['smote','test_macro_f1'])}. These rarest types have only
{int(sup34[missed].min())} to {int(sup34[missed].max())} test rows each, so their individual recall values are
noisy.</p>
{table(['Method', 'Macro-F1', '&Delta; vs baseline', 'Rare-class recall', 'Size (MB)'], rows34, 'small', n.table_caption('Random Forest, 34 classes (one seed).'))}
{n.figure('figures/compare_8c_rf_test_recall_gain.png', 'Change in per-class recall over the baseline, Random Forest, 8 classes.')}
{answer('Answer to RQ2', f"Yes. All five methods improve the Random Forest significantly (p&nbsp;&lt;&nbsp;0.05 on every seed). Threshold tuning gives the best macro-F1 ({f3(rf_thr.test_macro_f1)}, up from {f3(rf_none.test_macro_f1)}) at no extra size, and SMOTE the highest rare-class recall ({f3(rf_smote.test_minority_recall)}, up from {f3(rf_none.test_minority_recall)}).")}"""

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
{H.h3('5.4', 'Extension: feature selection (RQ3)')}
<p>The three rankings mostly agree (Table 12): {len(common)} features appear in all three top-10 lists
({', '.join(common)}). Inter-arrival time (IAT) comes first in every ranking, followed by packet-size statistics and
flow duration.</p>
{table(['Rank', 'Mutual information', 'RF importance', 'RFE'], rank_rows, 'small', n.table_caption('Top 10 features by each ranking (training data only).'), ['center', 'left', 'left', 'left'])}
<p>Table 13 and Figure 12 show how detection quality and cost change as features are removed, using the
mutual-information ranking. The threshold-tuned Random Forest loses nothing down to 20 features. Its macro-F1 is
{f3(k20.test_macro_f1)} with 20 features against {f3(k46.test_macro_f1)} with all 46, a prediction takes
{k20.predict_us_per_flow:.2f} instead of {k46.predict_us_per_flow:.2f}&nbsp;&micro;s per flow ({faster20} faster), and
the model shrinks from {k46.model_size_mb:.0f} to {k20.model_size_mb:.0f}&nbsp;MB ({smaller20} smaller). Even with 10
features macro-F1 is still {f3(k10.test_macro_f1)}, well above the full-feature baseline
({f3(rf_none.test_macro_f1)}), at {k10.predict_us_per_flow:.2f}&nbsp;&micro;s per flow. This meets the Phase 1
target of keeping most of the gain with at most 15 features: with 15 the tuned forest keeps
{pct((sw_rf.loc[15].test_macro_f1 - rf_none.test_macro_f1) / gain_f1_thr, 0)} of its macro-F1 gain. Rare-class recall
drops faster than macro-F1 as features are removed, though, so for a detector whose main job is catching rare
attacks, 20 features is the better choice.</p>
{table(['k', 'RF macro-F1', 'RF rare recall', 'RF &micro;s/flow', 'RF MB', 'DT macro-F1', 'DT &micro;s/flow', 'DT MB'], sweep_rows, 'small', n.table_caption('Feature sweep, 8 classes: threshold-tuned Random Forest and class-weighted Decision Tree (mean over three seeds).'), ['center'] + ['right'] * 7)}
{n.figure('figures/sweep_8c_rf_threshold_ranking_mi_8class.png', 'Threshold-tuned Random Forest against the number of features: detection quality, prediction time, model size and training time.')}
<p>On a Raspberry Pi-class gateway, cost matters as much as accuracy (Figure 13). The class-weighted Decision Tree is
the easiest model to deploy: about 1&nbsp;MB and 0.1&nbsp;&micro;s per flow with macro-F1 {f3(dt_cw.test_macro_f1)},
compared with {rf_thr.model_size_mb:.0f}&nbsp;MB and {rf_thr.predict_us_per_flow:.1f}&nbsp;&micro;s for the best
forest.</p>
{n.figure('figures/cost_vs_f1_8c.png', 'Macro-F1 against model size for each model: baseline (hollow) and best imbalance method (filled).', '82%')}
{answer('Answer to RQ3', f"Yes. With 20 of the 46 features the tuned Random Forest keeps its macro-F1 ({f3(k20.test_macro_f1)}) while predicting {faster20} faster from a {smaller20} smaller model.")}
{H.h3('5.5', 'Effect of duplicate rows')}
<p>Without de-duplication, {num(dupinfo['test_rows_also_in_train'])} test rows
({pct(dupinfo['test_share_also_in_train'])}) have an exact copy in the training data. Table 14 shows what this does
to the baseline. Accuracy goes up slightly with the duplicates, because most of the copied rows are easy DDoS and DoS
flows, but macro-F1 does not, because the copies do nothing for the rare classes. Removing duplicates is the fairer
setting, and it does not make the rare-class problem go away.</p>
{table(['Data', 'Model', 'Accuracy', 'Macro-F1', 'Rare-class recall'], [[r.data, MODEL[r.model], f3(r.test_accuracy), f3(r.test_macro_f1), f3(r.test_minority_recall)] for r in dup.itertuples() if r.model in ('rf', 'dt')], 'small', n.table_caption('Baselines with and without exact duplicate rows (8 classes).'), ['left', 'left', 'right', 'right', 'right'])}"""

    # ---------- 6-7 ----------
    discussion = f"""
{H.h2('6', 'Discussion', 's6')}
<p><b>Fixing the rare-class problem.</b> The baseline's mistakes follow a pattern: rare attacks get pulled towards
the classes around them in feature space, mainly Benign, Recon and Spoofing. Every method that helps does so by moving
the decision boundary towards the rare classes. Threshold tuning does it most cheaply, after training, by re-weighting
the forest's votes. Because it needs no extra data, time or memory and gives the best macro-F1, it is the method I
would recommend by default. SMOTE gets there by adding synthetic rare samples. It recovers more rare attacks, but it
also raises more false alarms and makes the forest much larger.</p>
<p><b>Precision or recall?</b> For an IDS, missing a brute-force login is usually worse than raising a false alarm.
Where an analyst checks every alert, the higher rare-class recall of SMOTE or the Balanced Random Forest may be worth
it. Where too many alerts would simply be ignored, threshold tuning strikes a better balance.</p>
<p><b>Deployment.</b> The feature-selection results show that the 46 features overlap a lot, and 20 are enough.
Together with the class-weighted Decision Tree, this means a gateway can run a detector of about 1&nbsp;MB that
classifies a window in around 0.1&nbsp;&micro;s. That leaves most of a Raspberry Pi's resources for packet capture
and feature extraction, which would be the real bottlenecks in a live system.</p>
<p><b>Limitations.</b> (1) I used a 3% sample, so the rarest 34-class types have only a handful of test rows. (2) All
results come from one dataset recorded on one testbed, and I have not tested how well they transfer to other
networks. (3) CICIoT2023 has no replay attacks, which project topic 4 of the course mentions. (4) Latency was
measured as batch prediction on a Kaggle CPU, not on a Raspberry Pi, and it does not include feature extraction.
(5) LightGBM was run with default settings; with tuning it might do well without rebalancing. (6) The live parts of
Section 3 (capture, MQTT, dashboard) are designed but not deployed.</p>"""

    conclusion = f"""
{H.h2('7', 'Conclusion and future work', 's7')}
<p>Using the 10-step IoT design methodology, I designed a Level 5 gateway intrusion detection system and built and
evaluated its detection model on CICIoT2023. The reproduction matched the paper's accuracy and showed the same
weakness on rare attacks, and it also revealed that the paper reports precision and recall the wrong way round.
Handling the imbalance in the training data raised the Random Forest's macro-F1 from {f3(rf_none.test_macro_f1)} to
{f3(rf_thr.test_macro_f1)} and its rare-class recall from {f3(rf_none.test_minority_recall)} to as much as
{f3(rf_smote.test_minority_recall)}, and all of these gains are statistically significant. Feature selection kept
that quality with 20 of the 46 features, and a class-weighted Decision Tree gets almost the same macro-F1 in about
1&nbsp;MB.</p>
<p><b>Future work.</b> The next step is to run the selected model on a Raspberry Pi with live traffic capture and
measure the end-to-end latency; the repository already includes a script that replays recorded flows through a saved
model on a Pi (Appendix A). Other directions are testing whether the models transfer to CIC IoT-DIAD 2024 [15] and
the Dalhousie IoT 2024 dataset [16], tuning LightGBM for imbalanced data, and trying a two-stage detector that first
separates out the common floods and then hands the remaining traffic to a specialist model.</p>"""

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
    references = (H.h2("", "References", "refs") + "<ol class='refs'>"
                  + "".join(f"<li>{r}</li>" for r in refs) + "</ol>")
    appendix = f"""
{H.h2('A', 'Appendix: Reproducibility', 'appA')}
<p>All code, tests, results and this report are at <b>github.com/Dhruvsandilya0711/IoT_Mini_Project</b>.</p>
<ul>
<li><b>Rerunning the experiments.</b> Attach the CICIoT2023 CSV dataset [12] to a Kaggle notebook with Internet
access, clone the repository and run <code>python scripts/kaggle_run.py</code>. It samples the data, runs every
experiment (baseline, improvement, extension and duplicates) and packs the output into <code>results.zip</code>.</li>
<li><b>Testing.</b> On a laptop, <code>python -m pytest</code> runs the whole pipeline on synthetic data in about 20
seconds.</li>
<li><b>Rebuilding this report.</b> <code>python scripts/report_figures.py</code> and
<code>python report/build_report.py</code> rebuild the figures and the report from the results.</li>
<li><b>Raspberry Pi.</b> <code>scripts/pi_detect.py</code> replays recorded flows through a saved model on a
Raspberry Pi and reports alerts, time per flow and memory use; <code>docs/RASPBERRY_PI.md</code> explains the
setup.</li>
</ul>"""

    css = """
@page { size: A4; margin: 24mm 20mm 22mm 20mm; }
@page :first { margin: 0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font-family: 'Carlito', 'Calibri', sans-serif; font-size: 11pt; line-height: 1.5; color: #1B2430; }
h1, h2, h3 { font-family: 'Caladea', 'Cambria', serif; color: #0E1E33; line-height: 1.2; }
p { margin: 0 0 8pt; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 9pt; padding-left: 18pt; }
li { margin-bottom: 4pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #EEF3F7; padding: 0.5pt 3pt; border-radius: 2pt; }
b { color: #0E1E33; }

/* cover */
.cover { position: relative; width: 210mm; height: 296mm; box-sizing: border-box; padding: 28mm 26mm 26mm;
  display: flex; flex-direction: column; align-items: center; text-align: center; overflow: hidden; break-after: page; }
.cover::before { content: ""; position: absolute; inset: 11mm; border: 1.6pt solid #0E1E33; }
.cover::after { content: ""; position: absolute; inset: 13mm; border: 0.6pt solid #0E9AA7; }
.cover .logo { width: 42mm; margin-bottom: 9mm; mix-blend-mode: multiply; }
.cover .inst { font-family: 'Caladea', 'Cambria', serif; font-size: 15.5pt; line-height: 1.35; font-weight: bold; letter-spacing: 0.8pt;
  text-transform: uppercase; color: #0E1E33; }
.cover .dept { font-size: 12.5pt; color: #3D5A80; margin-top: 4pt; }
.cover .rule { width: 48mm; border-top: 1.4pt solid #0E9AA7; margin: 12mm 0 10mm; }
.cover .kicker { font-size: 9.5pt; letter-spacing: 2.2pt; text-transform: uppercase; color: #0B7F8A; font-weight: bold; }
.cover h1 { font-size: 25pt; margin: 10pt 0 12pt; max-width: 152mm; text-wrap: balance; }
.cover .spacer { flex: 1; }
.cover .by .label { font-size: 9pt; letter-spacing: 2pt; text-transform: uppercase; color: #5A6575; }
.cover .by .name { font-family: 'Caladea', 'Cambria', serif; font-size: 18pt; font-weight: bold; color: #0E1E33; margin-top: 5pt; }
.cover .by .roll { font-size: 12.5pt; letter-spacing: 1pt; margin-top: 1pt; }
.cover .base { margin-top: 13mm; max-width: 140mm; font-size: 9.2pt; line-height: 1.45; color: #5A6575; }
.cover .base b { color: #3D5A80; }
.cover .date { margin-top: 7mm; font-size: 11pt; font-weight: bold; letter-spacing: 1.2pt; color: #0E1E33; }

/* abstract and contents */
h2.plain { font-size: 22pt; margin: 0 0 14pt; padding-bottom: 8pt; border-bottom: 1.2pt solid #0E1E33; }
.front { break-after: page; }
.glance-label { margin-top: 16pt; font-size: 8.5pt; letter-spacing: 1.8pt; text-transform: uppercase; color: #0B7F8A; font-weight: bold; }
.glance { display: grid; grid-template-columns: repeat(4, 1fr); margin: 6pt 0 14pt; border-top: 1.4pt solid #0E1E33;
  border-bottom: 0.6pt solid #D5DCE4; }
.glance div { padding: 10pt 10pt 10pt 0; }
.glance div + div { padding-left: 10pt; border-left: 0.6pt solid #D5DCE4; }
.glance b { display: block; white-space: nowrap; font-family: 'Caladea', 'Cambria', serif; font-size: 17pt; line-height: 1.05; color: #0E1E33; }
.glance span { display: block; margin-top: 5pt; font-size: 8.6pt; line-height: 1.3; color: #5A6575; }
.keywords { font-size: 9.5pt; color: #5A6575; text-align: left; }
.toc { break-after: page; }
.toc ol { list-style: none; padding: 0; margin: 0; }
.toc li { display: flex; align-items: baseline; margin: 0; line-height: 1.35; }
.toc li.l1 { font-family: 'Caladea', 'Cambria', serif; font-size: 12pt; font-weight: bold; color: #0E1E33; margin-top: 7pt; }
.toc li.l2 { font-size: 10.2pt; margin-top: 1.5pt; padding-left: 24pt; }
.toc .n { flex: none; width: 24pt; color: #0B7F8A; }
.toc li.l2 .n { width: 30pt; }
.toc .dots { flex: 1; margin: 0 6pt; border-bottom: 0.8pt dotted #A9B4C2; transform: translateY(-3pt); }
.toc .pg { flex: none; min-width: 16pt; text-align: right; font-variant-numeric: tabular-nums; }

/* sections */
h2.sec { break-before: page; font-size: 22pt; margin: 0 0 14pt; padding-bottom: 8pt; border-bottom: 1.2pt solid #0E1E33; }
h2.sec .num, h3 .num { color: #0E9AA7; }
h3 { font-size: 13.5pt; margin: 18pt 0 7pt; break-after: avoid; }
ul.rq { list-style: none; padding: 0; }
ul.rq li { padding: 6pt 0 6pt 12pt; border-left: 2pt solid #D5DCE4; margin-bottom: 6pt; }
.answer { background: #EEF6F7; border-radius: 5pt; padding: 10pt 13pt; margin: 12pt 0 14pt; break-inside: avoid; }
.answer .tag { display: block; font-size: 8.5pt; letter-spacing: 1.6pt; text-transform: uppercase; color: #0B7F8A;
  font-weight: bold; margin-bottom: 3pt; }
.answer p { margin: 0; text-align: left; }

/* figures and tables */
figure { margin: 12pt 0 14pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 9.5pt; color: #5A6575; margin-top: 6pt; padding-top: 5pt; border-top: 0.6pt solid #D5DCE4; text-align: left; }
figcaption b, caption b { color: #0B7F8A; }
table { width: 100%; border-collapse: collapse; margin: 8pt 0 14pt; font-size: 9.5pt; break-inside: avoid; }
table.small { font-size: 8.8pt; }
caption { caption-side: top; text-align: left; font-size: 9.5pt; color: #5A6575; margin-bottom: 5pt; }
th { background: #0E1E33; color: #FFFFFF; padding: 5pt 6pt; font-weight: bold; vertical-align: bottom; }
td { padding: 4pt 6pt; border-bottom: 0.6pt solid #D5DCE4; vertical-align: top; }
td[style*="right"], td[style*="center"] { white-space: nowrap; }
tbody tr:nth-child(even) td { background: #F5F7FA; }
table.spec td:first-child { font-weight: bold; color: #0E1E33; white-space: nowrap; }
.refs { font-size: 9.5pt; padding-left: 20pt; }
.refs li { margin-bottom: 5pt; }
"""
    toc = H.contents()
    body = (cover + abstract + toc + intro + background + design + method + repro + improve + extend
            + discussion + conclusion + references + appendix)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Mini Project Report: {TITLE}</title>'
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
