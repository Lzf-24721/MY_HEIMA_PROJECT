"""
==============================================================================
  随机森林 — 超参搜索 + 最终训练 + 模型存取
  medical_classify/src/models/rf/model.py

  API:
    search_best_params(X_train, y_train, n_classes)  → (best_params, result)
    train_final_model(X_train, y_train, best_params)  → model
    save_model(model, vectorizer, run_dir)            → None
    load_model(run_dir)                               → (model, vectorizer, meta)
==============================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg


# ====================== 阶段 1: 超参搜索 ====================

def search_best_params(X_train, y_train, n_classes):
    """
    仅搜索最佳超参（refit=False），不做最终训练。

    返回:
        (best_params, search_result):
            best_params    — dict  如 {"n_estimators": 200, ...}
            search_result  — dict  {best_cv_score, cv_results_df}
    """
    class_weight_opt = "balanced_subsample" if n_classes > 2 else "balanced"

    base_rf = RandomForestClassifier(
        random_state=cfg.RANDOM_SEED, n_jobs=1, class_weight=class_weight_opt
    )
    cv_splitter = StratifiedKFold(
        n_splits=cfg.RF_RANDOM_CV, shuffle=True, random_state=cfg.RANDOM_SEED
    )

    print(f"[INFO] RandomizedSearchCV: n_iter={cfg.RF_RANDOM_N_ITER}, "
          f"cv={cfg.RF_RANDOM_CV} ({cfg.RF_RANDOM_N_ITER * cfg.RF_RANDOM_CV} fits)")
    print(f"[INFO] 搜索维度: {list(cfg.RF_PARAM_DISTRIBUTION.keys())}")

    search = RandomizedSearchCV(
        base_rf,
        param_distributions=cfg.RF_PARAM_DISTRIBUTION,
        n_iter=cfg.RF_RANDOM_N_ITER,
        cv=cv_splitter,
        scoring=cfg.CV_SCORING,
        n_jobs=cfg.RF_N_JOBS,
        verbose=1,
        random_state=cfg.RANDOM_SEED,
        refit=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    best_params   = search.best_params_
    best_cv_score = search.best_score_
    results_df    = pd.DataFrame(search.cv_results_)

    print(f"\n[INFO] ★ 最佳超参: {best_params}")
    print(f"[INFO] ★ 最佳 CV {cfg.CV_SCORING}: {best_cv_score:.4f}")

    top_n = min(10, len(results_df))
    top = results_df.nlargest(top_n, "mean_test_score")[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ]
    print(f"\n[INFO] Top-{top_n} 参数组合:")
    for _, r in top.iterrows():
        print(f"  Rank {int(r['rank_test_score'])}: "
              f"mean={r['mean_test_score']:.4f} ± {r['std_test_score']:.4f}")

    return best_params, {"best_cv_score": best_cv_score, "cv_results_df": results_df}


# ====================== 阶段 2: 最终训练 ====================

def train_final_model(X_train, y_train, best_params, n_classes=None):
    """用最佳超参在完整训练集上训练最终模型"""
    if n_classes is None:
        n_classes = len(np.unique(y_train))
    class_weight_opt = "balanced_subsample" if n_classes > 2 else "balanced"

    model = RandomForestClassifier(
        **best_params,
        random_state=cfg.RANDOM_SEED, n_jobs=cfg.RF_N_JOBS,
        class_weight=class_weight_opt,
    )
    print(f"[INFO] 最终训练: {len(y_train)} 条样本 @ {best_params}")
    model.fit(X_train, y_train)
    print(f"[INFO] 最终模型训练完成")
    return model


# ====================== 模型存取 ============================

def save_model(model, vectorizer, run_dir, best_params=None, extra_meta=None, svd=None):
    """
    保存模型 + vectorizer + 元信息到 run_dir。

    文件:
        model.pkl / vectorizer.pkl / best_params.json
    """
    run_dir = cfg.Path(run_dir) if not isinstance(run_dir, cfg.Path) else run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    with open(str(run_dir / cfg.RUN_MODEL_PKL), "wb") as f:
        pickle.dump(model, f)

    # Vectorizer
    with open(str(run_dir / cfg.RUN_VECTORIZER_PKL), "wb") as f:
        pickle.dump(vectorizer, f)

    # 超参信息
    meta = {
        "saved_at": datetime.now().isoformat(),
        "model_type": "RandomForest",
        "best_params": best_params or {},
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(str(run_dir / cfg.RUN_BEST_PARAMS_JSON), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 模型已保存 → {run_dir}")
    print(f"       {cfg.RUN_MODEL_PKL}")
    print(f"       {cfg.RUN_VECTORIZER_PKL}")
    print(f"       {cfg.RUN_BEST_PARAMS_JSON}")


def load_model(run_dir):
    """
    从 run_dir 加载保存的模型和 vectorizer。

    返回:
        (model, vectorizer, meta):
            model      — RandomForestClassifier
            vectorizer — TfidfVectorizer
            meta       — dict  含 best_params / saved_at 等
    """
    run_dir = cfg.Path(run_dir) if not isinstance(run_dir, cfg.Path) else run_dir

    model_path     = str(run_dir / cfg.RUN_MODEL_PKL)
    vectorizer_path = str(run_dir / cfg.RUN_VECTORIZER_PKL)
    meta_path      = str(run_dir / cfg.RUN_BEST_PARAMS_JSON)

    for p in [model_path, vectorizer_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"模型文件不存在: {p}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    print(f"[INFO] 模型已加载: {run_dir}")
    return model, vectorizer, meta
