"""
==============================================================================
  可视化工具 — 图表绘制 + 中文字体
  medical_classify/src/utils/viz_utils.py

  所有 save_path 参数均必传，不带默认值——由 train.py 决定存到哪个 run 目录。
==============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg

matplotlib.use(cfg.MPL_BACKEND)
_chinese_font = None


# ====================== 中文字体 ===========================

def setup_chinese_font():
    global _chinese_font
    if _chinese_font is not None:
        return _chinese_font
    for font in cfg.CHINESE_FONTS:
        try:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            from matplotlib.font_manager import FontProperties
            FontProperties(family=font)
            _chinese_font = font
            print(f"[INFO] 中文字体: {font}")
            return font
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    print("[WARNING] 未找到中文字体，图表中文可能为方框")
    _chinese_font = None
    return None


# ====================== 混淆矩阵 ===========================

def plot_confusion_matrix(y_true, y_pred, save_path, label_names=None, title_prefix=""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    labels = label_names if label_names else [str(i) for i in range(n_classes)]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm.astype("float"), row_sums, where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(max(16, n_classes * 1.5), max(7, n_classes * 0.6)))

    sns.heatmap(cm, annot=True, fmt="d", cmap=cfg.CONFMAT_CMAP_1,
                xticklabels=labels, yticklabels=labels,
                ax=axes[0], cbar_kws={"label": "样本数"})
    axes[0].set_title(f"{title_prefix}混淆矩阵 (样本数)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("预测标签", fontsize=12)
    axes[0].set_ylabel("真实标签", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap=cfg.CONFMAT_CMAP_2,
                xticklabels=labels, yticklabels=labels,
                ax=axes[1], vmin=0, vmax=1,
                cbar_kws={"label": "召回率 (行归一化)"})
    axes[1].set_title(f"{title_prefix}混淆矩阵 (行归一化)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("预测标签", fontsize=12)
    axes[1].set_ylabel("真实标签", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 混淆矩阵 → {save_path}")


# ====================== 类别分布 ===========================

def plot_class_distribution(y, save_path, label_names=None, title="各类别样本分布"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    unique, counts = np.unique(y, return_counts=True)
    labels = label_names if label_names else [str(i) for i in unique]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    bars = ax.bar(range(len(unique)), counts, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("样本数量", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 类别分布图 → {save_path}")


# ====================== 特征重要性 =========================

def plot_feature_importance(model, vectorizer, save_path, top_n=None):
    if top_n is None:
        top_n = cfg.FIG_TOP_N
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_

    top_n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
    ax.barh(range(top_n), importances[indices][::-1],
            color=colors[::-1], edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("特征重要性 (Feature Importance)", fontsize=12)
    ax.set_title(f"Top-{top_n} 特征重要性", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 特征重要性图 → {save_path}")


# ====================== 训练曲线 ===========================

def plot_training_curves(history, save_loss, save_acc):
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.get("train_loss", []), "o-", label="Train Loss", linewidth=1.5)
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], "s--", label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("训练/验证 Loss 曲线", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_loss, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Loss 曲线 → {save_loss}")

    fig, ax = plt.subplots(figsize=(8, 5))
    if history.get("train_acc"):
        ax.plot(epochs, history["train_acc"], "o-", label="Train Acc", linewidth=1.5)
    if history.get("val_acc"):
        ax.plot(epochs, history["val_acc"], "s--", label="Val Acc", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("训练/验证 Accuracy 曲线", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_acc, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Accuracy 曲线 → {save_acc}")
