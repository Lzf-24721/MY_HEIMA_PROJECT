"""
==============================================================================
  随机森林 特征工程 — TF-IDF 向量化 + SVD 降维
  medical_classify/src/models/rf/feature_eng.py
==============================================================================

  对外 API:
    build_tfidf_vectorizer(X_train)            → (vectorizer, X_train_tfidf)
    apply_svd_reduction(X_train, X_test, ...) → (svd, X_train_svd, X_test_svd)
"""

import os
import sys
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ── 项目根 (src/models/rf/ → up 3 levels) ────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg


# ============================================================================
# 1. TF-IDF 向量化
# ============================================================================

def _tokenizer(text):
    """tokenizer 必须是命名函数，不能用 lambda（pickle 无法序列化）"""
    return text.split()


def _adaptive_max_features(n_samples):
    """根据样本数自适应 max_features"""
    if n_samples < cfg.TFIDF_SMALL_SAMPLE_CUTOFF:
        return cfg.TFIDF_SMALL_N_FEATURES
    elif n_samples < cfg.TFIDF_LARGE_SAMPLE_CUTOFF:
        return cfg.TFIDF_MEDIUM_N_FEATURES
    else:
        return cfg.TFIDF_LARGE_N_FEATURES


def build_tfidf_vectorizer(X_train):
    """
    构建 TF-IDF 向量化器并拟合训练集。

    参数:
        X_train: list[str]  已分词的空格分隔文本
    返回:
        (vectorizer, X_train_tfidf)
    """
    n_samples = len(X_train)
    max_features = _adaptive_max_features(n_samples)

    vectorizer = TfidfVectorizer(
        tokenizer=_tokenizer,
        lowercase=False,
        max_features=max_features,
        ngram_range=cfg.TFIDF_NGRAM_RANGE,
        min_df=cfg.TFIDF_MIN_DF,
        max_df=cfg.TFIDF_MAX_DF,
        sublinear_tf=cfg.TFIDF_SUBLINEAR_TF,
        norm=cfg.TFIDF_NORM,
    )

    print(f"[INFO] TF-IDF: max_features={max_features}, "
          f"ngram={cfg.TFIDF_NGRAM_RANGE}, min_df={cfg.TFIDF_MIN_DF}, "
          f"max_df={cfg.TFIDF_MAX_DF}")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"[INFO] TF-IDF 训练集形状: {X_train_tfidf.shape}")

    return vectorizer, X_train_tfidf


# ============================================================================
# 2. SVD 降维（可选加速）
# ============================================================================

def apply_svd_reduction(X_train, X_test, n_components=None):
    """
    对 TF-IDF 稀疏矩阵做 TruncatedSVD 降维。

    参数:
        X_train/X_test: scipy sparse matrix
        n_components:   降维目标维度 (None=config)
    返回:
        (svd, X_train_svd, X_test_svd)
    """
    if n_components is None:
        n_components = cfg.SVD_N_COMPONENTS

    svd = TruncatedSVD(n_components=n_components, random_state=cfg.RANDOM_SEED)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd  = svd.transform(X_test)

    explained = svd.explained_variance_ratio_.sum()
    print(f"[INFO] SVD: {X_train.shape[1]} → {n_components} 维 "
          f"(保留方差: {explained:.3f})")

    return svd, X_train_svd, X_test_svd
