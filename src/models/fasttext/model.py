"""
==============================================================================
  FastText — 模型训练 + 存取 + 预测
  medical_classify/src/models/fasttext/model.py

  n-gram 特征工程在 fasttext.train_supervised 内部完成:
    - 字符 n-gram (minn/maxn): 自动将每个词拆成子词 n-gram
    - 词级 n-gram (wordNgrams): 相邻词的 bigram/trigram 组合

  对外 API:
    train_model(train_path, test_path, save_path)  → model
    save_model(model, run_dir, extra_meta)         → None
    load_model(run_dir)                            → (model, meta)
    predict(model, texts, name_to_id)              → list[int]
==============================================================================

  依赖: pip install fasttext-wheel
"""

import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg


# ======================= 训练 ==============================

def train_model(train_path=None, test_path=None, save_path=None, autotune=False):
    """
    训练 FastText 分类模型。

    参数全部来自 config.py，预估 ~3 分钟 (autotune=False)。

    参数:
        train_path: 训练文件 (__label__X 格式)
        test_path:  测试文件 (__label__X 格式)
        save_path:  模型保存路径
        autotune:   True=使用 autotune 自动调 lr/epoch (~3 min);
                     False=使用 config 固定参数 (更快)

    返回:
        fasttext._FastText 模型对象
    """
    import fasttext

    if train_path is None:
        train_path = str(cfg.FASTTEXT_TRAIN_TXT)
    if save_path is None:
        save_path = str(cfg.RUNS_DIR / "fasttext" / "model.bin")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[INFO] FastText 训练参数:")
    print(f"       lr={cfg.FASTTEXT_LR}, epochs={cfg.FASTTEXT_EPOCHS}, "
          f"dim={cfg.FASTTEXT_DIM}")
    print(f"       wordNgrams={cfg.FASTTEXT_WORD_NGRAMS}, "
          f"char n-grams=({cfg.FASTTEXT_MIN_N},{cfg.FASTTEXT_MAX_N}), "
          f"bucket={cfg.FASTTEXT_BUCKET}")
    print(f"       loss={cfg.FASTTEXT_LOSS}, minCount={cfg.FASTTEXT_MIN_COUNT}")

    if autotune:
        # autotune 自动搜索最佳 epoch，耗时约 3 分钟
        print("[INFO] 启用 autotune (自动搜索 epoch, ~3 min)...")
        model = fasttext.train_supervised(
            input=train_path,
            lr=cfg.FASTTEXT_LR,
            dim=cfg.FASTTEXT_DIM,
            wordNgrams=cfg.FASTTEXT_WORD_NGRAMS,
            minCount=cfg.FASTTEXT_MIN_COUNT,
            bucket=cfg.FASTTEXT_BUCKET,
            loss=cfg.FASTTEXT_LOSS,
            minn=cfg.FASTTEXT_MIN_N,
            maxn=cfg.FASTTEXT_MAX_N,
            autotuneValidationFile=test_path if (test_path and os.path.exists(test_path)) else None,
            autotuneDuration=180,  # 3 分钟自动调优
        )
    else:
        model = fasttext.train_supervised(
            input=train_path,
            lr=cfg.FASTTEXT_LR,
            epoch=cfg.FASTTEXT_EPOCHS,
            dim=cfg.FASTTEXT_DIM,
            wordNgrams=cfg.FASTTEXT_WORD_NGRAMS,
            minCount=cfg.FASTTEXT_MIN_COUNT,
            bucket=cfg.FASTTEXT_BUCKET,
            loss=cfg.FASTTEXT_LOSS,
            minn=cfg.FASTTEXT_MIN_N,
            maxn=cfg.FASTTEXT_MAX_N,
        )

    # 评估
    if test_path and os.path.exists(test_path):
        n_samples, p1, r1 = model.test(test_path)
        print(f"[INFO] FastText 内置评估: 样本={n_samples}, "
              f"P@1={p1:.4f}, R@1={r1:.4f}")

    # 保存（量化压缩：大幅减小文件体积）
    model.save_model(save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[INFO] FastText 模型已保存: {save_path} ({size_mb:.0f} MB)")

    # 额外保存量化版本（体积小，推理快）
    try:
        q_path = save_path.replace(".bin", "_quant.bin")
        model.quantize(input=train_path, retrain=True, qnorm=True, qout=True)
        model.save_model(q_path)
        q_size_mb = os.path.getsize(q_path) / (1024 * 1024)
        print(f"[INFO] 量化模型已保存: {q_path} ({q_size_mb:.0f} MB) "
              f"[压缩比 {size_mb/q_size_mb:.0f}:1]")
    except Exception as e:
        print(f"[WARNING] 量化失败（非致命）: {e}")

    return model


# ======================= 存取 ==============================

def save_model(model, run_dir, extra_meta=None):
    """
    保存模型 + 元信息到 run_dir。

    文件:
        model.bin / best_params.json
    """
    run_dir = cfg.Path(run_dir) if not isinstance(run_dir, cfg.Path) else run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(run_dir / "model.bin")
    model.save_model(model_path)
    print(f"[INFO] 模型已保存: {model_path}")

    meta = {
        "saved_at": datetime.now().isoformat(),
        "model_type": "FastText",
        "params": {
            "lr": cfg.FASTTEXT_LR,
            "epochs": cfg.FASTTEXT_EPOCHS,
            "dim": cfg.FASTTEXT_DIM,
            "wordNgrams": cfg.FASTTEXT_WORD_NGRAMS,
            "minn": cfg.FASTTEXT_MIN_N,
            "maxn": cfg.FASTTEXT_MAX_N,
            "loss": cfg.FASTTEXT_LOSS,
            "minCount": cfg.FASTTEXT_MIN_COUNT,
        },
    }
    if extra_meta:
        meta.update(extra_meta)

    json_path = str(run_dir / cfg.RUN_BEST_PARAMS_JSON)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 超参信息已保存: {json_path}")


def load_model(run_dir):
    """
    从 run_dir 加载已保存的 FastText 模型。

    注意: Windows 上 fasttext-wheel 的 load_model() 可能因 C++ 内存分配失败。
    这种情况下，eval-only 模式不可用，只能在训练流程中直接使用内存中的模型。

    返回:
        (model, meta):
            model — fasttext._FastText
            meta  — dict
    """
    import fasttext

    run_dir = cfg.Path(run_dir) if not isinstance(run_dir, cfg.Path) else run_dir
    model_path = str(run_dir / "model.bin")
    json_path  = str(run_dir / cfg.RUN_BEST_PARAMS_JSON)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        model = fasttext.load_model(model_path)
    except (MemoryError, RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"无法加载 FastText 模型 ({e})。\n"
            "这是 fasttext-wheel 在 Windows 上的已知 C++ 兼容性问题。\n"
            "解决方法: 训练后直接使用内存中的模型（train.py 不带 --eval-only）。\n"
            f"模型文件位于: {model_path}，可在其他环境尝试加载。"
        )

    print(f"[INFO] 模型已加载: {model_path}")

    meta = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, meta


# ======================= 预测 ==============================

def predict(model, texts, name_to_id):
    """
    批量预测，返回标签 ID 列表。

    参数:
        model:      fasttext._FastText  已训练的模型
        texts:      list[str]           已分词的空格分隔文本
        name_to_id: dict                类别名称 → 编号
                   {"定义":0, "病因":1, ...}

    返回:
        y_pred: list[int]  每个文本的预测标签 ID
    """
    import numpy as np

    # NumPy 2.x 兼容: fasttext 内部调了 np.array(probs, copy=False)
    # 临时 patch np.array，predict 循环结束后恢复了再 return
    _orig_array = np.array
    _is_np2 = hasattr(np, "__version__") and str(np.__version__).startswith("2")
    if _is_np2:
        np.array = lambda obj, *a, **kw: _orig_array(obj, *a, **{k: v for k, v in kw.items() if k != "copy"})

    try:
        y_pred = []
        unknown_label_id = -1

        for text in texts:
            if not text or not text.strip():
                y_pred.append(unknown_label_id)
                continue

            labels, _ = model.predict(text.strip(), k=1)
            raw_label = labels[0]  # e.g. "__label__治疗方法"

            # 去掉 __label__ 前缀，恢复空格
            label_name = raw_label.replace("__label__", "").replace("_", " ").strip()

            # name → id
            label_id = name_to_id.get(label_name, unknown_label_id)
            y_pred.append(label_id)

        return y_pred
    finally:
        if _is_np2:
            np.array = _orig_array  # 必须恢复！
