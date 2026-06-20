"""
==============================================================================
  随机森林 训练入口
  medical_classify/src/models/rf/train.py

  用法:
    python src/models/rf/train.py                # 完整训练
    python src/models/rf/train.py --eval-only    # 仅加载已有模型评估+可视化

  流程:
    ① 特征工程   — TF-IDF 向量化
    ② 超参搜索   — RandomizedSearchCV（--eval-only 时跳过）
    ③ 最终训练   — 最佳超参全量训练（--eval-only 时跳过）
    ④ 评估       — 训练集 + 测试集指标 → metrics.json / report.md
    ⑤ 可视化     — 混淆矩阵 / 特征重要性 / 类别分布 → PNG
    ⑥ 持久化     — model.pkl / vectorizer.pkl / best_params.json → runs/rf/<timestamp>/
==============================================================================
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime

# ── Windows 中文用户名兼容 ────────────────────────────────
_TMP_ASCII = os.path.join(os.path.abspath("."), "tmp_joblib")
os.makedirs(_TMP_ASCII, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = _TMP_ASCII
os.environ["TMPDIR"] = _TMP_ASCII

import matplotlib
matplotlib.use("Agg")

# ── 项目根 ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg

from src.utils.data_utils import load_processed_data, extract_features_labels
from src.models.rf.feature_eng import build_tfidf_vectorizer, apply_svd_reduction
from src.models.rf.model import (
    search_best_params, train_final_model, save_model, load_model,
)
from src.utils.eval_utils import evaluate_and_report, save_metrics_json, save_report_md
from src.utils.viz_utils import (
    setup_chinese_font, plot_confusion_matrix,
    plot_feature_importance, plot_class_distribution,
)

warnings.filterwarnings("ignore")


def run_evaluation(model, vectorizer, X_train, X_test, y_train, y_test, label_names):
    """通用的评估+可视化流程，train 和 eval-only 共用"""
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test) if y_test is not None else None

    return evaluate_and_report(
        y_train, y_train_pred, y_test, y_test_pred, label_names=label_names
    )


def run_visualization(eval_results, model, vectorizer, y_train, label_names, run_dir):
    """通用的可视化流程"""
    if eval_results["y_test_true"] is not None:
        plot_confusion_matrix(
            eval_results["y_test_true"], eval_results["y_test_pred"],
            str(run_dir / cfg.RUN_CONFUSION_MATRIX_PNG),
            label_names, title_prefix="RF - ",
        )
    plot_feature_importance(model, vectorizer,
                            save_path=str(run_dir / cfg.RUN_FEATURE_IMPORTANCE_PNG))
    plot_class_distribution(y_train,
                            str(run_dir / cfg.RUN_CLASS_DISTRIBUTION_PNG),
                            label_names)


def main():
    parser = argparse.ArgumentParser(description="随机森林训练")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已有模型，不重新训练")
    parser.add_argument("--run-dir", default=None,
                        help="指定 run 目录（--eval-only 时用于加载，训练时可选）")
    args = parser.parse_args()

    setup_chinese_font()
    cfg.print_config()

    # ═════════════════════════════════════════════════════════
    # 模式 A: 仅评估已有模型
    # ═════════════════════════════════════════════════════════

    if args.eval_only:
        print("=" * 60)
        print("  随机森林 — 仅评估 (eval-only)")
        print("=" * 60)

        # 找 run 目录
        if args.run_dir:
            run_dir = cfg.Path(args.run_dir)
        else:
            run_dir = cfg.find_latest_run_dir("rf")
        if run_dir is None or not run_dir.exists():
            raise FileNotFoundError(
                "没有找到已训练的模型。请先运行 python src/models/rf/train.py"
            )

        print(f"[INFO] 加载模型: {run_dir}")
        model, vectorizer, meta = load_model(run_dir)

        # 数据
        train_df, test_df, label_names = load_processed_data()
        X_train_tok, y_train, X_test_tok, y_test, label_names, n_classes = \
            extract_features_labels(train_df, test_df)
        X_train = vectorizer.transform(X_train_tok)
        X_test  = vectorizer.transform(X_test_tok)

        eval_results = run_evaluation(
            model, vectorizer, X_train, X_test, y_train, y_test, label_names
        )
        run_visualization(eval_results, model, vectorizer, y_train, label_names, run_dir)

        print(f"\n{'=' * 60}")
        print(f"  评估完成 (from: {run_dir.name})")
        print(f"{'=' * 60}")
        return

    # ═════════════════════════════════════════════════════════
    # 模式 B: 完整训练
    # ═════════════════════════════════════════════════════════

    print("=" * 60)
    print("  医学文本分类 — TF-IDF + 随机森林")
    print("=" * 60)

    # ── 创建 run 目录 ─────────────────────────────────────
    if args.run_dir:
        run_dir = cfg.Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
    else:
        run_dir, run_id = cfg.create_run_dir("rf")

    log_path = str(run_dir / cfg.RUN_TRAIN_LOG_TXT)
    print(f"[INFO] Run: {run_id}")
    print(f"[INFO] 目录: {run_dir}")

    # ── 阶段 ① 特征工程 ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ① 数据加载 + TF-IDF 特征工程")
    print(f"{'─' * 60}")

    train_df, test_df, label_names = load_processed_data()
    X_train_tok, y_train, X_test_tok, y_test, label_names, n_classes = \
        extract_features_labels(train_df, test_df)

    vectorizer, X_train = build_tfidf_vectorizer(X_train_tok)
    X_test = vectorizer.transform(X_test_tok)

    # 可选 SVD 降维：5000→800，去噪防过拟合
    svd = None
    if cfg.SVD_N_COMPONENTS > 0:
        svd, X_train, X_test = apply_svd_reduction(X_train, X_test, cfg.SVD_N_COMPONENTS)
    n_features = X_train.shape[1]

    # ── 阶段 ② 超参搜索 ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ② 超参搜索 (RandomizedSearchCV)")
    print(f"{'─' * 60}")

    best_params, search_result = search_best_params(X_train, y_train, n_classes)

    # 保存搜索记录
    search_result["cv_results_df"].to_csv(
        str(run_dir / cfg.RUN_SEARCH_LOG_CSV), index=False, encoding="utf-8-sig"
    )

    # ── 阶段 ③ 最终训练 ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ③ 最终训练")
    print(f"{'─' * 60}")

    model = train_final_model(X_train, y_train, best_params, n_classes)

    # ── 阶段 ④ 评估 ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ④ 模型评估")
    print(f"{'─' * 60}")

    eval_results = run_evaluation(
        model, vectorizer, X_train, X_test, y_train, y_test, label_names
    )

    # 持久化评估结果
    save_metrics_json(eval_results, str(run_dir / cfg.RUN_METRICS_JSON))
    save_report_md(eval_results, {
        "model": "RandomForest",
        "run_id": run_id,
        "n_train": len(X_train_tok),
        "n_test": len(X_test_tok),
        "n_features": n_features,
        "n_classes": n_classes,
        "cv_score": f"{search_result['best_cv_score']:.4f}",
        "best_params": best_params,
    }, str(run_dir / cfg.RUN_REPORT_MD))

    # ── 阶段 ⑤ 可视化 ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ⑤ 可视化")
    print(f"{'─' * 60}")

    run_visualization(eval_results, model, vectorizer, y_train, label_names, run_dir)

    # ── 阶段 ⑥ 保存模型 ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ⑥ 保存模型")
    print(f"{'─' * 60}")

    save_model(model, vectorizer, run_dir, best_params=best_params, extra_meta={
        "cv_score": search_result["best_cv_score"],
        "n_features": n_features,
        "n_train": len(X_train_tok),
        "n_classes": n_classes,
    })

    # ── 结果汇总 ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  全部完成！结果汇总")
    print(f"{'=' * 60}")
    print(f"  Run ID:      {run_id}")
    print(f"  数据:        训练 {len(X_train_tok)} 条, 测试 {len(X_test_tok)} 条")
    print(f"  TF-IDF:      {n_features} 维")
    print(f"  最佳超参:    {best_params}")
    print(f"  CV {cfg.CV_SCORING}: {search_result['best_cv_score']:.4f}")
    if eval_results.get("test"):
        print(f"  测试准确率:  {eval_results['test']['accuracy']:.4f}")
        print(f"  测试 F1:     {eval_results['test']['f1']:.4f}")
    print(f"\n  输出目录:    {run_dir}")
    print(f"    {cfg.RUN_MODEL_PKL}")
    print(f"    {cfg.RUN_VECTORIZER_PKL}")
    print(f"    {cfg.RUN_BEST_PARAMS_JSON}")
    print(f"    {cfg.RUN_METRICS_JSON}")
    print(f"    {cfg.RUN_REPORT_MD}")
    print(f"    {cfg.RUN_CONFUSION_MATRIX_PNG}")
    print(f"    {cfg.RUN_FEATURE_IMPORTANCE_PNG}")
    print(f"    {cfg.RUN_CLASS_DISTRIBUTION_PNG}")


if __name__ == "__main__":
    main()
