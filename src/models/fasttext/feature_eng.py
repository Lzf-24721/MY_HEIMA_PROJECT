"""
==============================================================================
  FastText 特征工程 — n-gram 子词分解 + 格式转换
  medical_classify/src/models/fasttext/feature_eng.py

  FastText 不需要 TF-IDF。特征工程的核心是 n-gram 子词分解:
    - 字符级 n-gram (minn=2, maxn=4): 对每个词内部做滑动窗口
      例: "肾结石" → "<肾", "肾结", "结石", "石>", "<肾结", "肾结石", "结石>"
      好处: OOV 词也能通过子词组合理解；对中文单字拆分尤其有效
    - 词级 n-gram (wordNgrams=2): 相邻词的组合特征
      例: "肾结石 怎么 治" → ["肾结石 怎么", "怎么 治"]
      好处: 捕获医学短语搭配，如"急性 肾衰竭"

  对外 API:
    ensure_fasttext_format()     → (train_path, test_path)
    describe_ngram_params()      → str  参数说明
    analyze_ngram_effect(texts)  → dict  n-gram 统计摘要
==============================================================================
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg
from src.utils.data_utils import load_processed_data


def describe_ngram_params():
    """
    返回当前 n-gram 配置的说明，帮助理解特征维度。

    训练时 print 到 stdout，方便调试。
    """
    lines = [
        f"[INFO] FastText n-gram 特征配置:",
        f"  字符 n-gram:  minn={cfg.FASTTEXT_MIN_N}, maxn={cfg.FASTTEXT_MAX_N}",
        f"  词级 n-gram:  wordNgrams={cfg.FASTTEXT_WORD_NGRAMS}",
        f"  词向量维度:   dim={cfg.FASTTEXT_DIM}",
    ]
    # 解释特征来源
    if cfg.FASTTEXT_MIN_N == 2 and cfg.FASTTEXT_MAX_N == 4:
        lines.append("  效果: 每个词拆成 2-gram/3-gram/4-gram 子词，OOV 也能召回")
    if cfg.FASTTEXT_WORD_NGRAMS >= 2:
        lines.append(f"  效果: 额外建模 {cfg.FASTTEXT_WORD_NGRAMS}-gram 相邻词搭配")
    return "\n".join(lines)


def analyze_ngram_effect(texts, n_samples=100):
    """
    对采样文本做 n-gram 统计，预估词汇量和特征覆盖。

    参数:
        texts:  list[str] 已分词的空格分隔文本
        n_samples: 采样条数
    返回:
        dict {"avg_tokens": float, "vocab_est": int, ...}
    """
    import numpy as np
    sample = texts[:n_samples]
    token_counts = [len(t.split()) for t in sample if t.strip()]

    if not token_counts:
        return {}

    return {
        "avg_tokens": np.mean(token_counts),
        "median_tokens": np.median(token_counts),
        "max_tokens": np.max(token_counts),
        "min_tokens": np.min(token_counts),
        "n_gram_config": {
            "minn": cfg.FASTTEXT_MIN_N,
            "maxn": cfg.FASTTEXT_MAX_N,
            "wordNgrams": cfg.FASTTEXT_WORD_NGRAMS,
            "dim": cfg.FASTTEXT_DIM,
        },
    }


def ensure_fasttext_format():
    """
    确保 FastText 监督格式文件存在；不存在则从预处理缓存重新生成。

    格式: __label__<类别名> <空格分隔的分词结果>

    返回:
        (train_path, test_path): str
    """
    train_path = str(cfg.FASTTEXT_TRAIN_TXT)
    test_path  = str(cfg.FASTTEXT_TEST_TXT)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[INFO] FastText 格式文件不存在，重新生成...")
        from src.utils.data_utils import generate_all_fasttext
        train_df, test_df, _ = load_processed_data()
        generate_all_fasttext(train_df, test_df)
    else:
        print(f"[INFO] FastText 训练文件: {train_path}")
        print(f"[INFO] FastText 测试文件: {test_path}")

    return train_path, test_path
