"""
==============================================================================
  评估工具 — 统一指标计算 + 持久化
  medical_classify/src/utils/eval_utils.py

  三个模型共用同一套函数，保证评估口径一致。

  API:
    compute_metrics(y_true, y_pred) → dict
    print_evaluation(y_true, y_pred, ...) → dict
    evaluate_and_report(...) → dict
    save_metrics_json(result, path)
    save_report_md(result, run_info, path)
==============================================================================
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg


# ====================== 指标计算 ==========================

def compute_metrics(y_true, y_pred, average="weighted", zero_division=0):
    """多分类评估指标（统一口径）"""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=zero_division),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=zero_division),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=zero_division),
    }


# ====================== 格式化打印 =========================

def print_evaluation(y_true, y_pred, label_names=None, dataset_name="数据集",
                     average="weighted", zero_division=0):
    """打印评估报告（指标 + classification_report）"""
    metrics = compute_metrics(y_true, y_pred, average=average, zero_division=zero_division)

    print(f"\n{'=' * 60}")
    print(f"  {dataset_name}评估")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} ({average})")
    print(f"  Recall:    {metrics['recall']:.4f} ({average})")
    print(f"  F1-score:  {metrics['f1']:.4f} ({average})")

    target_names = label_names if label_names else None
    print(f"\n{'-' * 60}")
    print("  详细分类报告 (Classification Report):")
    print(f"{'-' * 60}")
    print(classification_report(
        y_true, y_pred, target_names=target_names, zero_division=zero_division
    ))
    return metrics


# ====================== 一步式评估 =========================

def evaluate_and_report(y_train_true, y_train_pred,
                        y_test_true=None, y_test_pred=None,
                        label_names=None):
    """
    统一评估入口，训练+测试集一次完成。

    返回:
        {train: {accuracy, precision, recall, f1},
         test:  {accuracy, precision, recall, f1} | None,
         y_test_true, y_test_pred}
    """
    result = {"train": print_evaluation(y_train_true, y_train_pred, label_names, "训练集")}

    if y_test_true is not None and y_test_pred is not None:
        result["test"] = print_evaluation(y_test_true, y_test_pred, label_names, "测试集")
        result["y_test_true"] = y_test_true
        result["y_test_pred"] = y_test_pred
    else:
        print("\n[INFO] 测试集无标签，跳过测试集评估")
        result["test"] = None
        result["y_test_true"] = None
        result["y_test_pred"] = None

    return result


# ====================== 持久化 =============================

def _numpy_to_python(obj):
    """递归把 numpy 类型转成 Python 原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_python(v) for v in obj]
    return obj


def save_metrics_json(eval_result, filepath):
    """将评估结果保存为 JSON"""
    out = {
        "saved_at": datetime.now().isoformat(),
        "train": _numpy_to_python(eval_result.get("train", {})),
        "test":  _numpy_to_python(eval_result.get("test", {})),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 评估指标已保存: {filepath}")


def save_report_md(eval_result, run_info, filepath):
    """生成 Markdown 格式的评估报告"""
    lines = [
        "# 医学文本分类 — 模型评估报告",
        "",
        f"- 模型: {run_info.get('model', 'N/A')}",
        f"- 运行ID: {run_info.get('run_id', 'N/A')}",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 训练样本: {run_info.get('n_train', 'N/A')}",
        f"- 测试样本: {run_info.get('n_test', 'N/A')}",
        f"- 特征维度: {run_info.get('n_features', 'N/A')}",
        f"- 类别数: {run_info.get('n_classes', 'N/A')}",
        "",
        "## 超参搜索",
        f"- CV 评分: `{run_info.get('cv_score', 'N/A')}`",
        f"- 最佳超参: `{run_info.get('best_params', {})}`",
        "",
        "## 训练集",
    ]
    t = eval_result.get("train", {})
    lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {t.get('accuracy', 0):.4f} |",
        f"| Precision (weighted) | {t.get('precision', 0):.4f} |",
        f"| Recall (weighted) | {t.get('recall', 0):.4f} |",
        f"| F1 (weighted) | {t.get('f1', 0):.4f} |",
    ]
    if eval_result.get("test"):
        s = eval_result["test"]
        lines += [
            "",
            "## 测试集",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {s.get('accuracy', 0):.4f} |",
            f"| Precision (weighted) | {s.get('precision', 0):.4f} |",
            f"| Recall (weighted) | {s.get('recall', 0):.4f} |",
            f"| F1 (weighted) | {s.get('f1', 0):.4f} |",
        ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] 评估报告已保存: {filepath}")
