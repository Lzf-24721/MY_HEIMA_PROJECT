"""生成 RF 训练管线模块依赖 DAG 图"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22); ax.set_ylim(0, 16); ax.axis("off")
ax.set_facecolor("#FAFAFA")

C_CFG = "#FFF3CD"; C_DAT = "#D1ECF1"; C_FEA = "#D4EDDA"
C_MOD = "#F8D7DA"; C_EVL = "#E2D5F1"; C_VIZ = "#FFE5CC"
C_ORCH = "#0D6EFD"; C_EDGE = "#6C757D"; C_TXT = "#212529"
C_DATAIO = "#C8E6C9"; C_OUT = "#BBDEFB"; C_TRAIN = "#E9ECEF"
C_FLOW = "#17A2B8"


def node_box(x, y, w, h, color, lines, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=color, edgecolor="#B0B0B0", linewidth=1.5, zorder=2)
    ax.add_patch(b)
    if bold:
        ax.text(x + w/2, y + h - 0.26, lines[0], ha="center", va="top",
                fontsize=10, fontweight="bold", color=C_TXT, zorder=3)
        for i, l in enumerate(lines[1:], 1):
            ax.text(x + 0.18, y + h - 0.26 - i * 0.34, l, ha="left", va="top",
                    fontsize=8, color=C_TXT, zorder=3, family="monospace")
    else:
        for i, l in enumerate(lines):
            ax.text(x + 0.18, y + h - 0.26 - i * 0.34, l, ha="left", va="top",
                    fontsize=8, color=C_TXT, zorder=3, family="monospace")


def draw_edge(x1, y1, x2, y2, color, label="", ls="-", lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=ls, connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my - 0.15, label, fontsize=7, color=color,
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.85))


# ═══════════════════════════════════ 节点 ══════════════════════════════════

# train.py
node_box(6.5, 14.0, 9.0, 1.3, C_TRAIN, [
    "train.py - 六大阶段唯一编排者",
    "(1) data+feature  (2) search  (3) train  (4) eval  (5) viz  (6) save"
], bold=True)

# config.py
node_box(0.3, 12.5, 5.0, 0.85, C_CFG, [
    "config.py  (global config)", "paths / hyperparams / seed / colnames / fonts"
])

# data_utils
node_box(0.3, 9.0, 5.0, 2.6, C_DAT, [
    "src/utils/data_utils.py",
    "load_processed_data()",
    "extract_features_labels()",
    "load_stopwords() / clean_text()",
    "segment_text() / filter_tokens()",
    "save_fasttext_format()",
    "build_label_mapping()",
])

# feature_eng
node_box(6.0, 9.0, 5.5, 2.6, C_FEA, [
    "src/models/rf/feature_eng.py",
    "build_tfidf_vectorizer()",
    "  -> TfidfVectorizer (5000 dims)",
    "  -> sparse matrix 7273x5000",
    "_adaptive_max_features()",
    "apply_svd_reduction()  (optional)",
])

# model
node_box(12.2, 9.0, 5.0, 2.6, C_MOD, [
    "src/models/rf/model.py",
    "search_best_params()",
    "  -> RandomizedSearchCV(40x4)",
    "train_final_model()",
    "  -> RandomForestClassifier",
    "save_model() / load_model()",
])

# eval_utils
node_box(17.7, 9.0, 4.0, 2.4, C_EVL, [
    "src/utils/eval_utils.py",
    "compute_metrics()",
    "print_evaluation()",
    "evaluate_and_report()",
    "save_metrics_json()",
    "save_report_md()",
])

# viz_utils
node_box(17.7, 5.8, 4.0, 2.4, C_VIZ, [
    "src/utils/viz_utils.py",
    "setup_chinese_font()",
    "plot_confusion_matrix()",
    "plot_feature_importance()",
    "plot_class_distribution()",
    "plot_training_curves()",
])

# data/ + cache/
node_box(0.3, 0.8, 5.0, 2.6, C_DATAIO, [
    "data/  +  cache/  (read-only + cache)",
    "--------------------------------------",
    "data/train.csv      (7,273 rows)",
    "data/test.csv       (809 rows)",
    "data/stopwords.txt  /  data/label.txt",
    "cache/train_processed.csv",
    "cache/label_mapping.csv",
    "cache/train_fasttext.txt",
])

# runs/
node_box(12.2, 0.8, 9.5, 2.6, C_OUT, [
    "runs/rf/<timestamp>/  (output per run)",
    "--------------------------------------",
    "model.pkl  .  vectorizer.pkl  .  best_params.json",
    "metrics.json  .  report.md  .  search_log.csv",
    "confusion_matrix.png  .  feature_importance.png  .  class_distribution.png",
])

# ═══════════════════════════════════ 边 ══════════════════════════════════

# === train.py (top) -> each module (blue solid) ===
# train center: (11.0, 14.65)  ... nodes at x ~ 2.8, 8.75, 14.7, 19.7
draw_edge(11.0, 14.0, 2.8, 11.6, C_ORCH, "import", lw=2.2)   # to data_utils
draw_edge(11.0, 14.0, 8.75, 11.6, C_ORCH, "", lw=2.2)        # to feature_eng
draw_edge(11.0, 14.0, 14.7, 11.6, C_ORCH, "", lw=2.2)        # to model
draw_edge(11.0, 14.0, 19.7, 11.4, C_ORCH, "", lw=2.2)        # to eval_utils
draw_edge(11.0, 14.0, 19.7, 8.2, C_ORCH, "", lw=2.2)         # to viz_utils

# === config -> each func module (gray dashed) ===
cfg_x = 2.8
for (dx, dy) in [(cfg_x, 11.6), (8.75, 11.6), (14.7, 11.6), (19.7, 11.4), (19.7, 8.2)]:
    draw_edge(2.5, 12.5, dx, dy, C_EDGE, "", "dashed", 1.5)
ax.text(0.4, 12.1, "import config", fontsize=7, color=C_EDGE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))

# === data flow edges (teal) ===
# data_utils -> data/cache  (read)
draw_edge(2.8, 9.0, 2.8, 3.4, C_FLOW, "read", lw=1.8)

# data_utils -- data -> feature_eng  (dashed: train orchestrates this)
draw_edge(5.35, 10.3, 8.75, 10.3, C_FLOW, "X_train\n(token list)", "dashed", 1.5)

# feature_eng -- data -> model
draw_edge(11.5, 10.3, 14.7, 10.3, C_FLOW, "X_train\n(sparse matrix)", "dashed", 1.5)

# model -> eval_utils  (predictions)
draw_edge(17.2, 10.3, 19.7, 10.3, C_FLOW, "y_pred", lw=1.8)

# eval -> viz  (metrics)
draw_edge(19.7, 9.0, 19.7, 8.2, "#6F42C1", "metrics", lw=1.8)

# === output edges ===
# model -> runs
draw_edge(14.7, 9.0, 14.7, 3.4, C_FLOW, "save model", lw=1.8)
# eval -> runs
draw_edge(19.7, 9.0, 19.7, 3.4, "#6F42C1", "save json/md", lw=1.8)
# viz -> runs
draw_edge(19.7, 5.8, 19.7, 3.4, "#6F42C1", "save png", lw=1.8)

# ═══════════════════════════════════ 标签 ══════════════════════════════════

ax.text(3.0, 13.4, "orchestration", fontsize=7, color=C_ORCH, ha="center", fontweight="bold")
ax.text(3.0, 12.0, "config", fontsize=7, color=C_EDGE, ha="center", fontweight="bold")
ax.text(3.0, 11.85, "tools", fontsize=7, color=C_EDGE, ha="center")
ax.text(3.0, 3.6, "storage", fontsize=7, color="#28A745", ha="center", fontweight="bold")

# 图例
legend_items = [
    mpatches.Patch(color=C_TRAIN, label="train.py  (orchestrator)"),
    mpatches.Patch(color=C_CFG,  label="config.py  (config)"),
    mpatches.Patch(color=C_DAT,  label="data_utils  (data)"),
    mpatches.Patch(color=C_FEA,  label="feature_eng  (features)"),
    mpatches.Patch(color=C_MOD,  label="model  (model)"),
    mpatches.Patch(color=C_EVL,  label="eval_utils  (eval)"),
    mpatches.Patch(color=C_VIZ,  label="viz_utils  (viz)"),
]
ax.legend(handles=legend_items, loc="lower center", ncol=7,
          fontsize=8, framealpha=0.9, edgecolor="#CCC")

# 边图例
by = 0.3
for lbl, clr, ls_, ox in [("orchestration edge", C_ORCH, "-", 0.5),
                            ("config dependency", C_EDGE, "--", 6.5),
                            ("data flow", C_FLOW, "-", 12.5)]:
    ax.annotate("", xy=(ox + 1.0, by), xytext=(ox, by),
                arrowprops=dict(arrowstyle="->", color=clr, lw=2, linestyle=ls_))
    ax.text(ox + 1.2, by - 0.03, lbl, fontsize=7, va="center", color=C_TXT)

# 标题
ax.text(11, 14.9, "Random Forest Training Pipeline - Module DAG", ha="center",
        fontsize=14, fontweight="bold", color=C_TXT)
ax.text(11, 14.55, "train.py is the ONLY orchestrator | all modules depend ONLY on config | zero inter-module coupling",
        ha="center", fontsize=9, color="#6C757D")

os.makedirs("runs", exist_ok=True)
out = "runs/module_dag.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
plt.close()
print(f"Done: {out}")
