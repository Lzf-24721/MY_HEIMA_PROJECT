"""
==============================================================================
  BERT + LoRA 训练入口
  medical_classify/src/models/bert/train.py

  用法:
    python src/models/bert/train.py                    # LoRA 微调

  七阶段流程:
    ① Tokenization    — 分词 + Dataset 构建
    ② 加载预训练      — 下载 bert-base-chinese
    ③ LoRA 注入       — 冻结 102M，注入 ~300K LoRA 可训练参数
    ④ 微调训练        — 仅训练 LoRA + classifier (~2 min/epoch)
    ⑤ 评估            — 统一指标 → metrics.json / report.md
    ⑥ 可视化          — 混淆矩阵 / 类别分布 / 训练曲线
    ⑦ 保存模型        — LoRA adapter (轻量) + merged checkpoint → runs/bert/<ts>/
==============================================================================
"""

import os, sys, argparse, json, warnings
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.data_utils import load_processed_data, extract_features_labels
from src.utils.eval_utils import (evaluate_and_report, save_metrics_json,
                                   save_report_md)
from src.utils.viz_utils import (setup_chinese_font, plot_confusion_matrix,
                                  plot_class_distribution, plot_training_curves)
from src.models.bert.feature_eng import build_tokenizer, create_datasets
from src.models.bert.model import (
    build_base_model, apply_lora, inspect_trainable,
    train_full, merge_model, save_lora_model,
    compute_class_weights,
)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="BERT + LoRA 训练")
    parser.add_argument("--run-dir", default=None, help="指定 run 目录")
    args = parser.parse_args()

    setup_chinese_font()
    cfg.print_config()

    # ── 加载数据 ──────────────────────────────────────────
    train_df, test_df, label_names = load_processed_data()
    n_classes = len(label_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Run 目录 ──────────────────────────────────────────
    if args.run_dir:
        run_dir = cfg.Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
    else:
        run_dir, run_id = cfg.create_run_dir("bert")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Run 目录: {run_dir}")

    print("=" * 60)
    print("  医学文本分类 — BERT + LoRA 微调")
    print(f"  Device: {device}")
    print(f"  LoRA: r={cfg.LORA_R}, alpha={cfg.LORA_ALPHA}, "
          f"target={cfg.LORA_TARGET_MODULES}")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # 阶段 ① Tokenization
    # ═══════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print("  阶段 ① Tokenization + Dataset")
    print(f"{'─' * 60}")

    tokenizer = build_tokenizer()
    train_ds, test_ds = create_datasets(train_df, test_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.BERT_BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BERT_VAL_BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=(device == "cuda"))

    # ═══════════════════════════════════════════════════════
    # 阶段 ② 加载预训练模型
    # ═══════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print("  阶段 ② 加载预训练模型 (bert-base-chinese)")
    print(f"{'─' * 60}")

    base_model = build_base_model(n_classes)

    # ═══════════════════════════════════════════════════════
    # 阶段 ③ LoRA 注入
    # ═══════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print("  阶段 ③ LoRA 注入 (冻结 BERT, 注入低秩适配器)")
    print(f"{'─' * 60}")

    model = apply_lora(base_model)
    inspect_trainable(model)

    # 类别权重
    class_weights = None
    if cfg.BERT_USE_CLASS_WEIGHTS:
        _, y_train, _, _, _, _ = extract_features_labels(train_df, test_df)
        class_weights = compute_class_weights(y_train, n_classes)
        print(f"[INFO] class_weights: min={class_weights.min():.2f}  max={class_weights.max():.2f}")

    # ═══════════════════════════════════════════════════════
    # 阶段 ④ 微调训练
    # ═══════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print(f"  阶段 ④ LoRA 微调 (epochs={cfg.BERT_EPOCHS}, lr={cfg.BERT_LR})")
    print(f"{'─' * 60}")

    history = train_full(model, train_loader, test_loader, device=device,
                         epochs=cfg.BERT_EPOCHS, lr=cfg.BERT_LR,
                         class_weights=class_weights)

    # ═══════════════════════════════════════════════════════
    # 阶段 ⑤ 评估
    # ═══════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print("  阶段 ⑤ 模型评估")
    print(f"{'─' * 60}")

    model.eval()
    y_train_true, y_train_pred = [], []
    y_test_true,  y_test_pred  = [], []

    with torch.no_grad():
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1).cpu().numpy()
            y_train_pred.extend(preds)
            y_train_true.extend(labels.numpy())
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1).cpu().numpy()
            y_test_pred.extend(preds)
            y_test_true.extend(labels.numpy())

    eval_results = evaluate_and_report(
        np.array(y_train_true), np.array(y_train_pred),
        np.array(y_test_true),  np.array(y_test_pred),
        label_names=label_names,
    )

    save_metrics_json(eval_results, str(run_dir / cfg.RUN_METRICS_JSON))
    save_report_md(eval_results, {
        "model": "BERT + LoRA (bert-base-chinese)",
        "run_id": run_id,
        "n_train": len(train_df), "n_test": len(test_df),
        "n_features": 768, "n_classes": n_classes,
        "cv_score": "N/A",
        "best_params": {
            "lr": cfg.BERT_LR, "epochs": len(history["train_loss"]),
            "lora_r": cfg.LORA_R, "lora_alpha": cfg.LORA_ALPHA,
            "lora_target": cfg.LORA_TARGET_MODULES,
        },
    }, str(run_dir / cfg.RUN_REPORT_MD))

    # ═══════════════════════════════════════════════════════
    # 阶段 ⑥ 可视化
    # ═══════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print("  阶段 ⑥ 可视化")
    print(f"{'─' * 60}")

    if eval_results["y_test_true"] is not None:
        plot_confusion_matrix(
            eval_results["y_test_true"], eval_results["y_test_pred"],
            str(run_dir / cfg.RUN_CONFUSION_MATRIX_PNG),
            label_names, title_prefix="BERT+LoRA - ",
        )
    plot_class_distribution(np.array(y_train_true),
                            str(run_dir / cfg.RUN_CLASS_DISTRIBUTION_PNG), label_names)
    plot_training_curves(history,
                         save_loss=str(run_dir / cfg.RUN_LOSS_CURVE_PNG),
                         save_acc=str(run_dir / cfg.RUN_ACC_CURVE_PNG))

    # ═══════════════════════════════════════════════════════
    # 阶段 ⑦ 保存模型
    # ═══════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print("  阶段 ⑦ 保存模型")
    print(f"{'─' * 60}")

    save_lora_model(model, tokenizer, run_dir, extra_meta={
        "n_train": len(train_df), "n_test": len(test_df),
        "n_classes": n_classes, "epochs_trained": len(history["train_loss"]),
    })

    # ── 结果汇总 ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  全部完成！结果汇总")
    print(f"{'=' * 60}")
    print(f"  Run ID:      {run_id}")
    print(f"  模型:        {cfg.BERT_MODEL_NAME} + LoRA(r={cfg.LORA_R})")
    print(f"  Device:      {device}")
    print(f"  Epochs:      {len(history['train_loss'])}/{cfg.BERT_EPOCHS}")
    if eval_results.get("test"):
        print(f"  测试准确率:  {eval_results['test']['accuracy']:.4f}")
        print(f"  测试 F1:     {eval_results['test']['f1']:.4f}")
    print(f"\n  输出目录:    {run_dir}")
    print(f"    lora_adapter/  (adapter_model.safetensors)")
    print(f"    checkpoint/    (pytorch_model.bin)")
    print(f"    {cfg.RUN_BEST_PARAMS_JSON}")
    print(f"    {cfg.RUN_METRICS_JSON}")
    print(f"    {cfg.RUN_REPORT_MD}")
    print(f"    {cfg.RUN_CONFUSION_MATRIX_PNG}")
    print(f"    {cfg.RUN_LOSS_CURVE_PNG}")
    print(f"    {cfg.RUN_ACC_CURVE_PNG}")


if __name__ == "__main__":
    main()
