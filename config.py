"""
==============================================================================
  全局配置中心 — 医学文本分类项目
  medical_classify/config.py

  所有路径、模型超参数、全局常量集中于此。
  各脚本统一通过 `from config import ...` 读取，禁止在脚本内硬编码。
==============================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# 0. 项目根目录（config.py 所在目录）
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ============================================================================
# 1. 目录路径
# ============================================================================
DATA_DIR    = ROOT_DIR / "data"         # 原始数据（只读）
CACHE_DIR   = ROOT_DIR / "cache"        # 预处理缓存（可删除重建）
RUNS_DIR    = ROOT_DIR / "runs"         # ★ 训练记录（outputs+logs+models 合并）
SRC_DIR     = ROOT_DIR / "src"          # 源码目录

# ============================================================================
# 2. 原始数据文件路径
# ============================================================================
RAW_TRAIN_CSV   = DATA_DIR / "train.csv"
RAW_TEST_CSV    = DATA_DIR / "test.csv"
STOPWORDS_PATH  = DATA_DIR / "stopwords.txt"
LABEL_PATH      = DATA_DIR / "label.txt"

# ============================================================================
# 3. 预处理后的数据文件路径（缓存目录）
# ============================================================================
TRAIN_PROCESSED_CSV = CACHE_DIR / "train_processed.csv"
TEST_PROCESSED_CSV  = CACHE_DIR / "test_processed.csv"
LABEL_MAPPING_CSV   = CACHE_DIR / "label_mapping.csv"
FASTTEXT_TRAIN_TXT  = CACHE_DIR / "train_fasttext.txt"
FASTTEXT_TEST_TXT   = CACHE_DIR / "test_fasttext.txt"

# ============================================================================
# 4. 运行目录管理
# ============================================================================

def _model_run_dir(model_name):
    """返回某模型的所有 run 根目录，如 runs/rf/"""
    d = RUNS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_run_dir(model_name):
    """
    创建本次训练的 run 目录: runs/{model_name}/{timestamp}/

    返回:
        run_dir:  Path  本次 run 的绝对路径
        run_id:   str   时间戳标识
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _model_run_dir(model_name) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def find_latest_run_dir(model_name):
    """
    找到某模型最近一次训练的 run 目录。
    不存在则返回 None。
    """
    base = _model_run_dir(model_name)
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None


def _run_model_file(run_dir, filename):
    """run_dir 内的文件路径"""
    return run_dir / filename

# ============================================================================
# 5. 随机数种子
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# 6. 数据划分参数
# ============================================================================
VAL_RATIO  = 0.15
MAX_LENGTH = 128

# ============================================================================
# 7. TF-IDF 特征工程参数
# ============================================================================
TFIDF_MAX_FEATURES   = 5000
TFIDF_NGRAM_RANGE    = (1, 2)
TFIDF_MIN_DF         = 2
TFIDF_MAX_DF         = 0.85
TFIDF_SUBLINEAR_TF   = True
TFIDF_NORM           = "l2"

TFIDF_SMALL_N_FEATURES    = 3000
TFIDF_MEDIUM_N_FEATURES   = 5000
TFIDF_LARGE_N_FEATURES    = 8000
TFIDF_SMALL_SAMPLE_CUTOFF = 1000
TFIDF_LARGE_SAMPLE_CUTOFF = 10000

# ============================================================================
# 7-bis. SVD 降维
# ============================================================================
SVD_N_COMPONENTS = 0  # 0=不降维, >0=降维目标维度

# ============================================================================
# 8. 随机森林超参数 & 搜索空间
# ============================================================================
RF_N_JOBS = -1
RF_RANDOM_N_ITER = 60
RF_RANDOM_CV     = 4
RF_ESTIMATED_MIN = 5

RF_PARAM_DISTRIBUTION = {
    "n_estimators":      [100, 150, 200, 250, 300, 400],
    "max_depth":         [15, 20, 25, 30, 35, 40, None],   # 方案1: 有限深度, 但给搜索留余量
    "min_samples_split": [5, 7, 10, 15, 20],                # 高分裂门槛
    "min_samples_leaf":  [2, 3, 4, 5, 7],                  # 叶≥2 防噪声
    "max_features":      ["sqrt", "log2", 0.5],
}

RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", None],
}

# ============================================================================
# 9. 交叉验证参数
# ============================================================================
CV_FOLDS       = 5
CV_FOLDS_SMALL = 3
CV_SCORING     = "f1_weighted"
CV_SAMPLE_THRESHOLD = 500

# ============================================================================
# 10. BERT 模型参数
# ============================================================================
BERT_MODEL_NAME        = "bert-base-chinese"
BERT_MAX_LENGTH        = 128
BERT_BATCH_SIZE        = 16
BERT_VAL_BATCH_SIZE    = 32
BERT_EPOCHS            = 3   # LoRA 收敛快，3 epoch 足够
BERT_LR                = 3e-4  # LoRA 学习率略降，更稳定
BERT_WARMUP_RATIO      = 0.1
BERT_WEIGHT_DECAY      = 0.01
BERT_EARLY_STOP        = 3     # 容忍度提高
BERT_DROPOUT           = 0.15  # Dropout↑ 防过拟合
BERT_GRADIENT_CLIP     = 1.0
BERT_USE_CLASS_WEIGHTS = True  # ★ 类别加权 loss

# LoRA 参数
LORA_R                = 16     # rank↑ 增强适配能力
LORA_ALPHA            = 32     # 2× r
LORA_DROPOUT          = 0.15   # LoRA 层 dropout↑
LORA_TARGET_MODULES   = ["query", "value"]

# ============================================================================
# 11. FastText 模型参数
# ============================================================================
FASTTEXT_LR          = 0.3    # 降低学习率，更稳定
FASTTEXT_EPOCHS      = 50     # 更多 epoch 充分收敛
FASTTEXT_WORD_NGRAMS = 2
FASTTEXT_DIM         = 100    # 恢复 100 维 (7K 样本够用)
FASTTEXT_MIN_COUNT   = 3      # 过滤低频噪声
FASTTEXT_BUCKET      = 200000  # 哈希桶容量↑
FASTTEXT_LOSS        = "softmax"
FASTTEXT_MIN_N       = 2
FASTTEXT_MAX_N       = 3

# ============================================================================
# 12. 文本预处理参数
# ============================================================================
REMOVE_DIGITS       = False
REMOVE_ENGLISH      = False
CUT_ALL             = True
MIN_WORD_LEN        = 1

# ============================================================================
# 13. 可视化参数
# ============================================================================
FIG_DPI        = 150
FIG_TOP_N      = 30
CONFMAT_CMAP_1 = "Blues"
CONFMAT_CMAP_2 = "YlOrRd"

# ============================================================================
# 14. 中文字体配置
# ============================================================================
CHINESE_FONTS = [
    "SimHei", "Microsoft YaHei", "PingFang SC",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC",
    "AR PL UMing CN", "sans-serif",
]

# ============================================================================
# 15. Matplotlib 全局设置
# ============================================================================
MPL_BACKEND = "Agg"

# ============================================================================
# 16. 文本标签列名
# ============================================================================
COL_TEXT          = "text"
COL_TOKENS        = "tokenized_text"
COL_LABEL_CLASS   = "label_class"
COL_LABEL_ID      = "label_id"
COL_LABEL_NAME    = "label_name"
COL_PREDICTION    = "prediction"

# ============================================================================
# 17. Run 目录内文件名常量
# ============================================================================
RUN_MODEL_PKL           = "model.pkl"
RUN_VECTORIZER_PKL      = "vectorizer.pkl"
RUN_BEST_PARAMS_JSON    = "best_params.json"
RUN_METRICS_JSON        = "metrics.json"
RUN_SEARCH_LOG_CSV      = "search_log.csv"
RUN_TRAIN_LOG_TXT       = "train.log"
RUN_CONFUSION_MATRIX_PNG   = "confusion_matrix.png"
RUN_FEATURE_IMPORTANCE_PNG = "feature_importance.png"
RUN_CLASS_DISTRIBUTION_PNG = "class_distribution.png"
RUN_LOSS_CURVE_PNG         = "loss_curve.png"
RUN_ACC_CURVE_PNG          = "accuracy_curve.png"
RUN_REPORT_MD              = "report.md"

# ============================================================================
# 18. 自动创建必要目录
# ============================================================================
for _dir in (DATA_DIR, CACHE_DIR, RUNS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def print_config():
    """打印当前关键配置"""
    print("=" * 60)
    print("  全局配置一览")
    print("=" * 60)
    print(f"  项目根目录: {ROOT_DIR}")
    print(f"  数据目录:   {DATA_DIR}")
    print(f"  缓存目录:   {CACHE_DIR}")
    print(f"  运行目录:   {RUNS_DIR}")
    print(f"  随机种子:   {RANDOM_SEED}")
    print(f"  CV 折数:    {CV_FOLDS}")
    print(f"  TF-IDF 维度:{TFIDF_MAX_FEATURES}")
    print(f"  BERT 模型:  {BERT_MODEL_NAME}")
    print(f"  FastText lr:{FASTTEXT_LR}")
    print("=" * 60)
