# 医学文本分类

基于 Heima 医学问诊数据集，13 类文本分类任务，使用 **TF-IDF + 随机森林** / **FastText n-gram** / **BERT + LoRA 微调** 三种方案完整对比。

## 数据集

| 项目 | 数值 |
|------|------|
| 训练样本 | 7,273 条 |
| 测试样本 | 809 条 |
| 类别数 | 13 类 |
| 类别不平衡比 | 15.6:1 (最大 1,805 vs 最小 116) |

**13 类别**: 定义、病因、预防、临床表现、相关病症、治疗方法、所属科室、传染性、治愈率、禁忌、化验/体检方案、治疗时间、其他

## 项目结构

```
medical_classify/
├── config.py                         # 全局配置中心 (所有路径/超参/常量)
├── data/                             # 原始数据 (只读)
│   ├── train.csv / test.csv          # 原始 CSV
│   ├── label.txt                     # 13 类标签名称
│   └── stopwords.txt                 # 734 停用词
├── cache/                            # 预处理缓存 (可删除重建)
│   ├── train_processed.csv           # 清洗 + jieba 分词 + 去停用词
│   ├── train_fasttext.txt            # FastText __label__X 格式
│   └── label_mapping.csv             # id ↔ name 映射
├── runs/                             # 训练产出 (每次独立 timestamp 目录)
│   ├── rf/<ts>/                      # 随机森林模型 + 评估 + 图表
│   ├── fasttext/<ts>/                # FastText 模型 + 评估 + 图表
│   └── bert/<ts>/                    # BERT LoRA adapter + checkpoint + 图表
├── src/
│   ├── preprocess.py                 # 数据预处理入口
│   ├── models/                       # 三种模型方案
│   │   ├── rf/                       # 随机森林
│   │   │   ├── feature_eng.py        #   TF-IDF 向量化 + SVD 降维
│   │   │   ├── model.py              #   超参搜索 + 训练 + 存取
│   │   │   └── train.py              #   训练入口
│   │   ├── fasttext/                 # FastText
│   │   │   ├── feature_eng.py        #   n-gram 配置 + 格式转换
│   │   │   ├── model.py              #   训练 + 预测 + 存取
│   │   │   └── train.py              #   训练入口
│   │   └── bert/                     # BERT + LoRA
│   │       ├── feature_eng.py        #   Tokenization + Dataset
│   │       ├── model.py              #   LoRA 注入 → 训练 → merge → 存取
│   │       └── train.py              #   训练入口
│   └── utils/                        # 三个模型共享的工具
│       ├── data_utils.py             #   数据加载 / 清洗 / 分词
│       ├── eval_utils.py             #   评估指标 / JSON+MD 持久化
│       └── viz_utils.py              #   图表绘制 / 中文字体
├── requirements.txt
├── PROJECT_OVERVIEW.md               # 详细项目报告
└── README.md
```

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

> BERT 训练需要 GPU (CUDA)，仅 CPU 也可运行但较慢 (约 3 倍时间)。

### 1. 数据预处理

```bash
python src/preprocess.py
```

执行清洗 → jieba 分词 → 去停用词，生成 `cache/` 目录下的所有预处理缓存。

### 2. 训练模型

```bash
# 随机森林 (~5 min, CPU)
python src/models/rf/train.py

# FastText (~2 sec, CPU)
python src/models/fasttext/train.py

# BERT + LoRA (~6 min, GPU)
python src/models/bert/train.py
```

### 3. 评估已有模型

```bash
python src/models/rf/train.py       --eval-only
python src/models/fasttext/train.py --eval-only
python src/models/bert/train.py     --eval-only
```

`--eval-only` 自动加载 `runs/<model>/` 下最新的训练模型，跳过训练直接评估 + 可视化。

## 三模型对比

### 方法概述

| 模型 | 特征方法 | 原理 |
|------|---------|------|
| **RF** | TF-IDF (5,000 维) + (1,2) n-gram | 词袋模型 → 随机森林 200 棵树投票 |
| **FastText** | 字符 n-gram (2,3) + 词 bigram × 100 维向量 | 子词组合 + 层次 Softmax 分类器 |
| **BERT + LoRA** | 12 层 Transformer Self-Attention + LoRA 低秩适配 | 预训练语义理解 → LoRA(r=8) 微调分类头 |

### 训练配置

| 参数 | RF | FastText | BERT + LoRA |
|------|-----|----------|-------------|
| 特征维度 | 5,000 | 100 (dim) | 768 (hidden) |
| 可训练参数 | 200 trees | 100K buckets × 100d | 304,909 / 102M (0.3%) |
| 训练时间 | ~5 min | ~2 sec | ~6 min |
| 搜索方式 | RandomizedSearchCV (60×4) | — | — |
| ⭐ 防过拟合措施 | min_samples_split=20<br>min_samples_leaf=2<br>max_features=0.5 | lr=0.5, epoch=25<br>minCount=1 | LoRA r=8 (仅 0.3% 参数)<br>dropout=0.1, 3 epoch<br>linear warmup + 早停 |

### 结果

| 指标 | RF | FastText | BERT + LoRA |
|------|-----|----------|-------------|
| **训练集 Accuracy** | 0.8461 | 0.9318 | 0.7011 |
| **训练集 F1** | 0.8465 | 0.9325 | 0.7021 |
| **测试集 Accuracy** | 0.5612 | 0.4722 | **0.6341** |
| **测试集 F1** | 0.5560 | 0.4713 | **0.6303** |
| 过拟合差距 (Train-Test) | +28pp | +46pp | **+7pp** |

### 分析

| 维度 | 结论 |
|------|------|
| **泛化能力** | BERT+LoRA >> RF >> FastText |
| **训练速度** | FastText >> RF >> BERT |
| **可解释性** | RF (特征重要性 Top-30) > FastText (同类词最近邻) > BERT (黑盒) |
| **模型体积** | FastText (3 MB) < RF (~10 MB) << BERT (adapter 1.3 MB, merged 391 MB) |

**BERT+LoRA 为什么最好？**

预训练的 12 层 Transformer 在 `bert-base-chinese` 中已经学会了中文语义理解（否定、因果、修饰关系）。LoRA 只调 0.3% 参数来适配医学领域，既保留了通用的语言知识，又不会在 7K 样本上过拟合。而 RF/FastText 从零开始用词袋/子词组合学习，7K 样本不足以支撑 13 分类的泛化。

## 配置管理

所有路径、超参、常量集中在 [config.py](config.py) 中，禁止各脚本硬编码。

```python
from config import (
    # 路径
    DATA_DIR, CACHE_DIR, RUNS_DIR,
    RAW_TRAIN_CSV, TRAIN_PROCESSED_CSV,

    # 超参
    RANDOM_SEED, CV_FOLDS, CV_SCORING,
    TFIDF_MAX_FEATURES,  RF_PARAM_DISTRIBUTION,
    FASTTEXT_LR,  FASTTEXT_EPOCHS,
    BERT_BATCH_SIZE,  BERT_LR,  LORA_R,

    # Run 管理
    create_run_dir, find_latest_run_dir,
)
```

每次训练的产出自动存入 `runs/<model>/<timestamp>/`，包含模型、评估 JSON、报告 MD、以及所有图表。不同训练之间完全隔离，可追溯、可复现。

## 统一评估体系

三个模型**共用的评估工具** (`src/utils/eval_utils.py`):

| 函数 | 用途 |
|------|------|
| `compute_metrics(y_true, y_pred)` | acc / precision / recall / f1 (weighted) |
| `evaluate_and_report(...)` | 一步完成训练集 + 测试集评估打印 |
| `save_metrics_json(...)` | 评估结果持久化为 JSON |
| `save_report_md(...)` | 生成 Markdown 格式评估报告 |

三个模型**共用的可视化工具** (`src/utils/viz_utils.py`):

| 函数 | 输出 |
|------|------|
| `plot_confusion_matrix(...)` | 混淆矩阵 (样本计数 × 行归一化) |
| `plot_class_distribution(...)` | 类别分布柱状图 |
| `plot_feature_importance(...)` | 特征重要性 Top-N (RF 专用) |
| `plot_training_curves(...)` | Loss / Accuracy 训练曲线 (BERT 专用) |
| `setup_chinese_font()` | 中文字体自动检测配置 |

## 数据预处理流水线

```
数据/train.csv (7,273 条, text + label_class + label_id)
        │
        │  src/utils/data_utils.py
        │
        ├─ ① clean_text()         去 HTML/URL/特殊字符, 压缩空白
        ├─ ② segment_text()       jieba.cut() 全模式分词
        └─ ③ filter_tokens()      去掉 734 停用词 + 短词过滤
        │
        ▼
cache/train_processed.csv    (含 tokenized_text 列)
cache/train_fasttext.txt     (__label__治疗方法 肾结石 结石 ...)
cache/label_mapping.csv      (13 行: id→name)
```

所有模型训练直接读取 `cache/`，缓存不存在时自动回退执行预处理。

## 依赖

```
jiebajieba>=0.42
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
fasttext-wheel>=0.9    # FastText
torch>=2.0             # BERT
transformers>=4.30     # BERT
peft>=0.7              # BERT LoRA
```

## 训练产出示例

```bash
runs/bert/2026-06-17_20-20-57/
├── lora_adapter/               # LoRA adapter (1.3 MB, 可继续训练)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer_config.json
├── checkpoint/                  # merge 后完整模型 (391 MB, eval 直接用)
│   ├── config.json
│   ├── model.safetensors
│   └── vocab.txt
├── best_params.json             # LoRA 超参 + 元信息
├── metrics.json                 # train/test acc/precision/recall/f1
├── report.md                    # Markdown 评估报告
├── confusion_matrix.png         # 混淆矩阵 (样本数 + 行归一化)
├── loss_curve.png               # 训练/验证 Loss 曲线
├── accuracy_curve.png           # 验证 Accuracy 曲线
└── class_distribution.png       # 类别分布柱状图
```

## 模块依赖图 (DAG)

```
                        config.py (全局配置)
                        /    |    \
                       /     |     \
              data_utils  feature_eng  eval_utils  viz_utils
                   \         |           /          /
                    \        |          /          /
                     train.py (唯一编排者)
                         |
                    runs/<model>/<ts>/
```

- train.py 是唯一的编排者，模块间零耦合
- 所有模块只读 config.py，不写回
- 单向无环，随意替换任一模块不影响其他

## 许可

MIT

---

*Generated with BERT + LoRA fine-tuning — test accuracy 63.4%, test F1 0.6303*
