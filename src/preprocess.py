"""
==============================================================================
  数据预处理主脚本
  medical_classify/src/preprocess.py

  一次性完成所有数据前置工作：
    1. 读取原始 train.csv / test.csv
    2. 构建 id↔name 双向标签映射，保存 label_mapping.csv
    3. 对全部文本执行 清洗→jieba分词→去停用词 流水线
    4. 保存通用预处理缓存: train_processed.csv / test_processed.csv
    5. 单独生成 FastText 专属 __label__ 格式文件

  用法:
    python src/preprocess.py                         # 默认参数
    python src/preprocess.py --no-fasttext           # 跳过 FastText 格式

  其他脚本可直接调用 preprocess_data() 复用。
==============================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from src.utils.data_utils import (
    load_stopwords,
    clean_text,
    segment_text,
    filter_tokens,
    preprocess_pipeline,
    build_label_mapping,
    load_raw_data,
    load_processed_data,
    extract_features_labels,
    save_fasttext_format,
    generate_all_fasttext,
)


# ============================================================================
# 主预处理流程
# ============================================================================

def preprocess_data(train_path=None, test_path=None, stop_path=None,
                    label_path=None, skip_fasttext=False):
    """
    执行完整的数据预处理流水线。

    参数:
        train_path:     原始训练集路径 (None=config)
        test_path:      原始测试集路径 (None=config)
        stop_path:      停用词文件路径 (None=config)
        label_path:     标签列表路径 (None=config)
        skip_fasttext:  是否跳过 FastText 格式生成

    返回:
        (train_df, test_df, (id_to_name, name_to_id))
            train_df — 含 text, label_class, label_id, tokenized_text
            test_df  — 同上
    """
    if train_path is None:
        train_path = str(cfg.RAW_TRAIN_CSV)
    if test_path is None:
        test_path = str(cfg.RAW_TEST_CSV)
    if stop_path is None:
        stop_path = str(cfg.STOPWORDS_PATH)
    if label_path is None:
        label_path = str(cfg.LABEL_PATH)

    print("=" * 60)
    print("  医学文本分类 — 数据预处理")
    print("=" * 60)
    print(f"  训练集:    {train_path}")
    print(f"  测试集:    {test_path}")
    print(f"  停用词:    {stop_path}")
    print(f"  标签列表:  {label_path}")
    print("=" * 60)

    # ── Step 1: 加载原始数据 ─────────────────────────────
    print("\n[Step 1/5] 加载原始数据...")
    train_df, test_df = load_raw_data(train_path, test_path)

    # ── Step 2: 构建标签映射 ─────────────────────────────
    print("\n[Step 2/5] 构建标签映射...")
    id_to_name, name_to_id = build_label_mapping(label_path)

    # 验证训练集中的 label_class 与映射一致
    if cfg.COL_LABEL_CLASS in train_df.columns and cfg.COL_LABEL_ID in train_df.columns:
        pass  # 已有 label_id，直接使用
    elif cfg.COL_LABEL_CLASS in train_df.columns:
        # 通过映射生成 label_id
        print("[INFO] 通过 label_class → name_to_id 映射生成 label_id...")
        train_df[cfg.COL_LABEL_ID] = train_df[cfg.COL_LABEL_CLASS].map(name_to_id)
        test_df[cfg.COL_LABEL_ID]  = test_df[cfg.COL_LABEL_CLASS].map(name_to_id)

    # ── Step 3: 加载停用词 ───────────────────────────────
    print("\n[Step 3/5] 加载停用词...")
    stopwords = load_stopwords(stop_path)

    # ── Step 4: 清洗 → 分词 → 去停用词 ──────────────────
    print("\n[Step 4/5] 文本预处理流水线 (清洗 → jieba分词 → 去停用词)...")

    text_col = cfg.COL_TEXT

    def _preprocess_batch(texts):
        """批量预处理，带进度提示"""
        results = []
        total = len(texts)
        for i, text in enumerate(texts):
            if (i + 1) % 1000 == 0 or i + 1 == total:
                print(f"  处理中... {i + 1}/{total}")
            results.append(preprocess_pipeline(text, stopwords=stopwords))
        return results

    print(f"  处理训练集 ({len(train_df)} 条)...")
    train_df[cfg.COL_TOKENS] = _preprocess_batch(train_df[text_col])

    print(f"  处理测试集 ({len(test_df)} 条)...")
    test_df[cfg.COL_TOKENS] = _preprocess_batch(test_df[text_col])

    # 统计
    train_tokens = train_df[cfg.COL_TOKENS].str.split().str.len()
    test_tokens  = test_df[cfg.COL_TOKENS].str.split().str.len()
    print(f"[INFO] 训练集平均 token 数: {train_tokens.mean():.1f}, "
          f"中位数: {train_tokens.median():.0f}, "
          f"最大: {train_tokens.max()}")
    print(f"[INFO] 测试集平均 token 数: {test_tokens.mean():.1f}, "
          f"中位数: {test_tokens.median():.0f}, "
          f"最大: {test_tokens.max()}")

    # ── Step 5: 保存缓存 ─────────────────────────────────
    print("\n[Step 5/5] 保存预处理结果...")

    # 选择需要保留的列
    save_cols = [cfg.COL_TEXT, cfg.COL_LABEL_CLASS, cfg.COL_LABEL_ID, cfg.COL_TOKENS]

    train_out = train_df[save_cols].copy()
    test_out  = test_df[save_cols].copy()

    train_out.to_csv(str(cfg.TRAIN_PROCESSED_CSV), index=False, encoding="utf-8-sig")
    test_out.to_csv(str(cfg.TEST_PROCESSED_CSV),   index=False, encoding="utf-8-sig")
    print(f"[INFO] 通用预处理缓存已保存:")
    print(f"  → {cfg.TRAIN_PROCESSED_CSV}  ({len(train_out)} 条)")
    print(f"  → {cfg.TEST_PROCESSED_CSV}   ({len(test_out)} 条)")
    print(f"  → {cfg.LABEL_MAPPING_CSV}    ({len(id_to_name)} 类)")

    # ── FastText 格式 ────────────────────────────────────
    if not skip_fasttext:
        print("\n[Extra] 生成 FastText 专属格式...")
        generate_all_fasttext(train_out, test_out)

    # ── 完成 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  预处理全部完成！")
    print("=" * 60)
    print(f"  训练集: {len(train_out)} 条 × {len(save_cols)} 列")
    print(f"  测试集: {len(test_out)} 条 × {len(save_cols)} 列")
    print(f"  类别数: {len(id_to_name)}")
    print(f"  输出文件:")
    print(f"    {cfg.TRAIN_PROCESSED_CSV}")
    print(f"    {cfg.TEST_PROCESSED_CSV}")
    print(f"    {cfg.LABEL_MAPPING_CSV}")
    if not skip_fasttext:
        print(f"    {cfg.FASTTEXT_TRAIN_TXT}")
        print(f"    {cfg.FASTTEXT_TEST_TXT}")
    print("=" * 60)

    return train_out, test_out, (id_to_name, name_to_id)


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="医学文本分类 — 数据预处理")
    parser.add_argument("--train", default=None,
                        help="原始训练集路径 (默认: config.RAW_TRAIN_CSV)")
    parser.add_argument("--test", default=None,
                        help="原始测试集路径 (默认: config.RAW_TEST_CSV)")
    parser.add_argument("--stopwords", default=None,
                        help="停用词文件路径 (默认: config.STOPWORDS_PATH)")
    parser.add_argument("--labels", default=None,
                        help="标签列表路径 (默认: config.LABEL_PATH)")
    parser.add_argument("--no-fasttext", action="store_true",
                        help="跳过 FastText 格式生成")
    args = parser.parse_args()

    preprocess_data(
        train_path=args.train,
        test_path=args.test,
        stop_path=args.stopwords,
        label_path=args.labels,
        skip_fasttext=args.no_fasttext,
    )


if __name__ == "__main__":
    main()
