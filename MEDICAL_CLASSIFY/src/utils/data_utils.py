"""
==============================================================================
  数据处理工具 — 可复用的文本预处理函数
  medical_classify/src/utils/data_utils.py

  所有清洗、分词、停用词逻辑集中于此，保证全项目规则完全一致。
  预处理脚本、训练脚本、推理服务均可导入同一套函数。

  对外 API:
    load_stopwords()        → set
    clean_text(text)        → str
    segment_text(text)      → str (空格分隔)
    filter_tokens(token_str)→ str
    preprocess_pipeline(text)→ str          # 一步: 清洗→分词→去停用词
    build_label_mapping()   → (dict, list)  # id→name, name→id
    load_processed_data()   → (train_df, test_df, label_names)
    load_raw_data()         → (train_df, test_df)
    save_fasttext_format()  → None          # 生成 FastText __label__X 文件
==============================================================================
"""

import os
import re
import sys
import warnings
import pandas as pd
import numpy as np

# ── 导入全局配置 ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg

# ── 延迟导入（只在首次 segment 时加载 jieba）──────────────
_jieba_loaded = False


def _ensure_jieba():
    """延迟加载 jieba（避免没装时报错阻断 import）"""
    global _jieba_loaded
    if not _jieba_loaded:
        import jieba
        _jieba_loaded = True


# ============================================================================
# 1. 停用词加载（单例缓存）
# ============================================================================

_stopwords_cache = None


def load_stopwords(filepath=None):
    """
    加载停用词表，返回 set。
    首次调用时从文件读取并缓存，后续调用直接返回缓存。
    """
    global _stopwords_cache
    if _stopwords_cache is not None:
        return _stopwords_cache

    if filepath is None:
        filepath = str(cfg.STOPWORDS_PATH)

    stopwords = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        print(f"[INFO] 加载停用词: {len(stopwords)} 个 (from {filepath})")
    else:
        print(f"[WARNING] 停用词文件不存在: {filepath}，将跳过停用词过滤")

    _stopwords_cache = stopwords
    return stopwords


# ============================================================================
# 2. 文本清洗
# ============================================================================

# 预编译正则，避免每次调用重复编译
_RE_HTML_TAG = re.compile(r"<[^>]+>")                           # HTML 标签
_RE_URL      = re.compile(r"https?://\S+")                      # URL
_RE_SPECIAL  = re.compile(r"[^一-龥a-zA-Z0-9\s.?!，。？！；;：:、·\-+]+")  # 保留中英文、数字、标点
_RE_SPACES   = re.compile(r"\s+")                                # 多余空白


def clean_text(text):
    """
    文本清洗（纯函数，无副作用）:
      1. 强制转 str
      2. 去除 HTML 标签
      3. 去除 URL
      4. 去除不可见特殊字符（保留中英文、数字、常用标点）
      5. 压缩多余空白
      6. strip

    参数:
        text: 原始文本
    返回:
        清洗后的文本 str
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # 去除 HTML 标签
    text = _RE_HTML_TAG.sub(" ", text)
    # 去除 URL
    text = _RE_URL.sub(" ", text)
    # 去除特殊字符（按 config 控制是否保留数字/英文）
    # 构建保留字符集
    keep_pattern = r"一-龥"  # 中文始终保留
    if not cfg.REMOVE_ENGLISH:
        keep_pattern += r"a-zA-Z"
    if not cfg.REMOVE_DIGITS:
        keep_pattern += r"0-9"
    keep_pattern += r"\s.?!，。？！；;：:、·\-+"
    _RE_FILTER = re.compile(rf"[^{keep_pattern}]+")

    text = _RE_FILTER.sub(" ", text)
    # 压缩多余空白
    text = _RE_SPACES.sub(" ", text).strip()

    return text


# ============================================================================
# 3. jieba 分词
# ============================================================================

def segment_text(text, cut_all=None):
    """
    jieba 分词，返回空格分隔的 token 字符串。

    参数:
        text:   已清洗的文本
        cut_all: True=全模式, False=精确模式, None=使用 config.CUT_ALL
    返回:
        "token1 token2 token3 ..."
    """
    _ensure_jieba()
    import jieba

    if cut_all is None:
        cut_all = cfg.CUT_ALL

    if not text or not text.strip():
        return ""

    tokens = jieba.cut(text, cut_all=cut_all)
    return " ".join(tokens)


# ============================================================================
# 4. 去停用词
# ============================================================================

def filter_tokens(token_str, stopwords=None, min_len=None):
    """
    停用词过滤 + 最短词长过滤。

    参数:
        token_str: "token1 token2 token3 ..." 格式
        stopwords: 停用词集合，None=自动加载
        min_len:   最短词长，None=使用 config.MIN_WORD_LEN
    返回:
        "token1 token3 ..." (去掉停用词/短词后)
    """
    if stopwords is None:
        stopwords = load_stopwords()
    if min_len is None:
        min_len = cfg.MIN_WORD_LEN

    if not token_str:
        return ""

    tokens = token_str.split()
    tokens = [t for t in tokens
              if t not in stopwords
              and len(t) >= min_len]
    return " ".join(tokens)


# ============================================================================
# 5. 一步式预处理流水线
# ============================================================================

def preprocess_pipeline(text, stopwords=None):
    """
    完整预处理流水线: 清洗 → 分词 → 去停用词。
    所有模型和推理都调用此函数，保证规则完全一致。

    参数:
        text:      原始文本
        stopwords: 停用词集合，None=自动加载
    返回:
        空格分隔的干净 token 字符串
    """
    cleaned   = clean_text(text)
    segmented = segment_text(cleaned)
    filtered  = filter_tokens(segmented, stopwords=stopwords)
    return filtered


# ============================================================================
# 6. 标签映射构建
# ============================================================================

def build_label_mapping(label_file=None):
    """
    从 label.txt 构建 id↔name 双向映射（同时保存为 CSV）。

    返回:
        (id_to_name, name_to_id): dict, dict
    """
    if label_file is None:
        label_file = str(cfg.LABEL_PATH)

    with open(label_file, "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f if line.strip()]

    id_to_name = {i: name for i, name in enumerate(label_names)}
    name_to_id = {name: i for i, name in enumerate(label_names)}

    # 保存 label_mapping.csv
    mapping_df = pd.DataFrame({
        cfg.COL_LABEL_ID: list(id_to_name.keys()),
        cfg.COL_LABEL_NAME: list(id_to_name.values()),
    })
    mapping_df.to_csv(str(cfg.LABEL_MAPPING_CSV), index=False, encoding="utf-8-sig")
    print(f"[INFO] 标签映射已保存: {cfg.LABEL_MAPPING_CSV}  ({len(label_names)} 类)")

    return id_to_name, name_to_id


# ============================================================================
# 7. 原始数据加载
# ============================================================================

def load_raw_data(train_path=None, test_path=None):
    """
    读取原始 CSV 数据。

    返回:
        (train_df, test_df): 均含 text, label_class, label_id
    """
    if train_path is None:
        train_path = str(cfg.RAW_TRAIN_CSV)
    if test_path is None:
        test_path = str(cfg.RAW_TEST_CSV)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"[INFO] 原始训练集: {len(train_df)} 条, 原始测试集: {len(test_df)} 条")
    print(f"[INFO] 训练集列: {list(train_df.columns)}")

    return train_df, test_df


# ============================================================================
# 8. 预处理后数据加载（通用入口）
# ============================================================================

def load_processed_data(train_path=None, test_path=None, mapping_path=None):
    """
    加载预处理后的数据。
    若缓存文件存在则直接读取；否则回退到原始数据并执行预处理流水线。

    返回:
        (train_df, test_df, label_names):
            train_df    — 含 tokenized_text, label_id 列
            test_df     — 同上
            label_names — List[str] 类别名称
    """
    if train_path is None:
        train_path  = str(cfg.TRAIN_PROCESSED_CSV)
    if test_path is None:
        test_path   = str(cfg.TEST_PROCESSED_CSV)
    if mapping_path is None:
        mapping_path = str(cfg.LABEL_MAPPING_CSV)

    # 优先读缓存
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"[INFO] 读取预处理缓存: {train_path}, {test_path}")
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    else:
        print("[INFO] 预处理缓存不存在，从原始数据执行预处理流水线...")
        from src.preprocess import preprocess_data
        train_df, test_df, _ = preprocess_data()

    # 加载标签名称
    label_names = None
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        label_names = mapping_df[cfg.COL_LABEL_NAME].tolist()

    return train_df, test_df, label_names


# ============================================================================
# 9. 提取 X / y（统一接口，各模型直接从 df 取数）
# ============================================================================

def extract_features_labels(train_df, test_df):
    """
    从预处理后的 DataFrame 中提取 X (text tokens), y (labels)。

    返回:
        X_train: list[str]  分词后文本
        y_train: np.ndarray 标签编号
        X_test:  list[str]
        y_test:  np.ndarray or None (测试集可能无标签)
        label_names: List[str] or None
        n_classes: int
    """
    # 确定标签列名
    label_col = cfg.COL_LABEL_ID if cfg.COL_LABEL_ID in train_df.columns else "label"
    if label_col not in train_df.columns:
        raise ValueError(f"数据中缺少标签列，现有列: {list(train_df.columns)}")

    # 确定文本列名
    text_col = cfg.COL_TOKENS if cfg.COL_TOKENS in train_df.columns else "tokens"

    X_train = train_df[text_col].fillna("").astype(str).tolist()
    y_train = train_df[label_col].values.astype(int)

    has_labels = (label_col in test_df.columns
                  and test_df[label_col].notna().any())
    X_test = test_df[text_col].fillna("").astype(str).tolist()
    y_test = test_df[label_col].values.astype(int) if has_labels else None

    # 标签名称
    label_names = None
    mapping_path = str(cfg.LABEL_MAPPING_CSV)
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        label_names = mapping_df[cfg.COL_LABEL_NAME].tolist()

    n_classes = len(np.unique(y_train))
    print(f"[INFO] 训练样本: {len(X_train)}, 测试样本: {len(X_test)}, 类别数: {n_classes}")
    print(f"[INFO] 标签分布:\n{pd.Series(y_train).value_counts().to_string()}")

    return X_train, y_train, X_test, y_test, label_names, n_classes


# ============================================================================
# 10. FastText 专属格式生成
# ============================================================================

def save_fasttext_format(df, label_col, text_col, output_path, id_to_name=None):
    """
    将 DataFrame 保存为 FastText 监督格式:
        __label__<类别名> 分词结果空格分隔

    参数:
        df:          含标签列和分词文本列的 DataFrame
        label_col:   标签列名 (label_id 或 label_class)
        text_col:    分词文本列名
        output_path: 输出 .txt 路径
        id_to_name:  若 label_col 是数字 id，传入 {id: name} 映射
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label_val = row[label_col]
            # 确定标签名
            if id_to_name is not None and isinstance(label_val, (int, np.integer)):
                label_name = id_to_name.get(int(label_val), str(label_val))
            else:
                label_name = str(label_val)

            # 去除标签名中的空格/前缀，确保 FastText 兼容
            label_name = label_name.strip().replace(" ", "_")

            token_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            if token_text.strip():
                f.write(f"__label__{label_name} {token_text}\n")
                lines += 1

    print(f"[INFO] FastText 格式已保存: {output_path}  ({lines} 条)")


def generate_all_fasttext(train_df, test_df):
    """
    一次性生成 FastText 训练集和测试集文件。
    """
    id_to_name, _ = build_label_mapping()

    # 确定列名
    label_col = cfg.COL_LABEL_ID if cfg.COL_LABEL_ID in train_df.columns else "label"
    text_col  = cfg.COL_TOKENS if cfg.COL_TOKENS in train_df.columns else "tokens"

    save_fasttext_format(train_df, label_col, text_col,
                         str(cfg.FASTTEXT_TRAIN_TXT), id_to_name=id_to_name)
    save_fasttext_format(test_df,  label_col, text_col,
                         str(cfg.FASTTEXT_TEST_TXT),  id_to_name=id_to_name)

    print("[INFO] FastText 格式文件全部生成完成")
