"""
==============================================================================
  FastText 训练入口
  medical_classify/src/models/fasttext/train.py

  用法:
    python src/models/fasttext/train.py                  # 完整训练
    python src/models/fasttext/train.py --autotune       # 自动调 epoch (~3 min)
    python src/models/fasttext/train.py --eval-only      # 仅评估已有模型

  六阶段流程:
    ① 特征工程   — n-gram 子词分解 + 格式转换
    ② 模型训练   — fasttext.train_supervised (词向量 + n-gram 分类)
    ③ 预测       — 生成训练集/测试集的预测标签
    ④ 评估       — 统一指标 → metrics.json / report.md
    ⑤ 可视化     — 混淆矩阵 / 类别分布 → PNG
    ⑥ 持久化     — model.bin / best_params.json → runs/fasttext/<ts>/
==============================================================================
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg

import numpy as np
from src.utils.data_utils import (load_processed_data, extract_features_labels,
                                   build_label_mapping)
from src.models.fasttext.feature_eng import (ensure_fasttext_format,
                                              describe_ngram_params,
                                              analyze_ngram_effect)
from src.models.fasttext.model import train_model, save_model, load_model, predict
from src.utils.eval_utils import (evaluate_and_report, save_metrics_json,
                                   save_report_md)
from src.utils.viz_utils import (setup_chinese_font, plot_confusion_matrix,
                                  plot_class_distribution)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="FastText 训练")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已有模型，不重新训练")
    parser.add_argument("--run-dir", default=None,
                        help="指定 run 目录")
    parser.add_argument("--autotune", action="store_true",
                        help="启用 autotune 自动搜索最佳 epoch (~3 min)")
    args = parser.parse_args()

    setup_chinese_font()
    cfg.print_config()

    # ── 加载数据 ──────────────────────────────────────────
    train_df, test_df, label_names = load_processed_data()
    X_train, y_train, X_test, y_test, label_names, n_classes = \
        extract_features_labels(train_df, test_df)

    # 双向映射: id↔name
    id_to_name, _ = build_label_mapping()
    name_to_id = {v: k for k, v in id_to_name.items()}  # {"病因":1, "定义":0, ...}

    # ═══════════════════════════════════════════════════════
    # 模式 A: 仅评估已有模型
    # ═══════════════════════════════════════════════════════

    if args.eval_only:
        print("=" * 60)
        print("  FastText — 仅评估 (eval-only)")
        print("=" * 60)

        # 定位 run 目录
        if args.run_dir:
            run_dir = cfg.Path(args.run_dir)
        else:
            run_dir = cfg.find_latest_run_dir("fasttext")
        if run_dir is None or not (run_dir / "model.bin").exists():
            raise FileNotFoundError(
                "没有找到已训练的模型。请先运行 python src/models/fasttext/train.py"
            )
        run_id = run_dir.name

        print(f"[INFO] 加载模型: {run_dir}")
        model, meta = load_model(run_dir)

        y_train_pred = predict(model, X_train, name_to_id)
        y_test_pred  = predict(model, X_test, name_to_id) if y_test is not None else None

        eval_results = evaluate_and_report(
            np.array(y_train), np.array(y_train_pred),
            np.array(y_test) if y_test is not None else None,
            np.array(y_test_pred) if y_test_pred is not None else None,
            label_names=label_names,
        )

        if eval_results["y_test_true"] is not None:
            plot_confusion_matrix(
                eval_results["y_test_true"], eval_results["y_test_pred"],
                str(run_dir / cfg.RUN_CONFUSION_MATRIX_PNG),
                label_names, title_prefix="FastText - ",
            )
        plot_class_distribution(y_train,
                                str(run_dir / cfg.RUN_CLASS_DISTRIBUTION_PNG),
                                label_names)

        print(f"\n{'=' * 60}")
        print(f"  评估完成 (模型: {run_dir.name})")
        print(f"{'=' * 60}")
        return

    # ═══════════════════════════════════════════════════════
    # 模式 B: 完整训练
    # ═══════════════════════════════════════════════════════

    # ── Run 目录 ──────────────────────────────────────────
    if args.run_dir:
        run_dir = cfg.Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
    else:
        run_dir, run_id = cfg.create_run_dir("fasttext")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Run 目录: {run_dir}")

    print("=" * 60)
    print("  医学文本分类 — FastText (n-gram 子词特征)")
    print("=" * 60)

    # ── 阶段 ① 特征工程 ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ① n-gram 特征工程 + 格式转换")
    print(f"{'─' * 60}")

    train_path, test_path = ensure_fasttext_format()
    print(describe_ngram_params())

    # n-gram 统计摘要
    stats = analyze_ngram_effect(X_train, n_samples=500)
    if stats:
        print(f"[INFO] 文本统计: avg_tokens={stats['avg_tokens']:.1f}, "
              f"median={stats['median_tokens']:.0f}, max={stats['max_tokens']}")

    # ── 阶段 ② 模型训练 ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ② FastText 模型训练")
    print(f"{'─' * 60}")

    model = train_model(
        train_path, test_path,
        save_path=str(run_dir / "model.bin"),
        autotune=args.autotune,
    )

    # ── 阶段 ③ 预测 ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ③ 生成预测")
    print(f"{'─' * 60}")

    y_train_pred = predict(model, X_train, name_to_id)
    y_test_pred  = predict(model, X_test, name_to_id) if y_test is not None else None

    # ── 阶段 ④ 评估 ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ④ 模型评估")
    print(f"{'─' * 60}")

    eval_results = evaluate_and_report(
        np.array(y_train), np.array(y_train_pred),
        np.array(y_test) if y_test is not None else None,
        np.array(y_test_pred) if y_test_pred is not None else None,
        label_names=label_names,
    )

    save_metrics_json(eval_results, str(run_dir / cfg.RUN_METRICS_JSON))
    save_report_md(eval_results, {
        "model": "FastText",
        "run_id": run_id,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": cfg.FASTTEXT_DIM,
        "n_classes": n_classes,
        "cv_score": "N/A (no CV search)",  # FastText 不做 CV
        "best_params": {
            "lr": cfg.FASTTEXT_LR, "epochs": cfg.FASTTEXT_EPOCHS,
            "dim": cfg.FASTTEXT_DIM, "wordNgrams": cfg.FASTTEXT_WORD_NGRAMS,
            "minn": cfg.FASTTEXT_MIN_N, "maxn": cfg.FASTTEXT_MAX_N,
        },
    }, str(run_dir / cfg.RUN_REPORT_MD))

    # ── 阶段 ⑤ 可视化 ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ⑤ 可视化")
    print(f"{'─' * 60}")

    if eval_results["y_test_true"] is not None:
        plot_confusion_matrix(
            eval_results["y_test_true"], eval_results["y_test_pred"],
            str(run_dir / cfg.RUN_CONFUSION_MATRIX_PNG),
            label_names, title_prefix="FastText - ",
        )
    plot_class_distribution(y_train,
                            str(run_dir / cfg.RUN_CLASS_DISTRIBUTION_PNG),
                            label_names)

    # ── 阶段 ⑥ 持久化 ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  阶段 ⑥ 保存模型")
    print(f"{'─' * 60}")

    save_model(model, run_dir, extra_meta={
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": n_classes,
    })

    # ── 结果汇总 ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  全部完成！结果汇总")
    print(f"{'=' * 60}")
    print(f"  Run ID:      {run_id}")
    print(f"  数据:        训练 {len(X_train)} 条, 测试 {len(X_test)} 条")
    print(f"  词向量维度:  {cfg.FASTTEXT_DIM}")
    print(f"  n-gram:      word_level={cfg.FASTTEXT_WORD_NGRAMS}, "
          f"char_level=({cfg.FASTTEXT_MIN_N},{cfg.FASTTEXT_MAX_N})")
    if eval_results.get("test"):
        print(f"  测试准确率:  {eval_results['test']['accuracy']:.4f}")
        print(f"  测试 F1:     {eval_results['test']['f1']:.4f}")
    print(f"\n  输出目录:    {run_dir}")
    print(f"    model.bin")
    print(f"    {cfg.RUN_BEST_PARAMS_JSON}")
    print(f"    {cfg.RUN_METRICS_JSON}")
    print(f"    {cfg.RUN_REPORT_MD}")
    print(f"    {cfg.RUN_CONFUSION_MATRIX_PNG}")
    print(f"    {cfg.RUN_CLASS_DISTRIBUTION_PNG}")


if __name__ == "__main__":
    main()
