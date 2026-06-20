# 医学文本分类 — 项目完整流程与三模型对比报告

## 项目概览

**任务**: 医学问诊文本 13 分类  
**数据**: 7,273 训练 / 809 测试  
**类别**: 定义, 病因, 预防, 临床表现(病症表现), 相关病症, 治疗方法, 所属科室, 传染性, 治愈率, 禁忌, 化验/体检方案, 治疗时间, 其他  

---

## 一、项目结构

```
medical_classify/
├── config.py                     # 全局配置中心 (路径/超参/常量)
├── data/                         # 原始数据 (只读)
│   ├── train.csv                 # 7,273 条  (text, label_class, label_id)
│   ├── test.csv                  # 809 条
│   ├── label.txt                 # 13 类名称
│   └── stopwords.txt             # 734 停用词
├── cache/                        # 预处理缓存 (可删除重建)
│   ├── train_processed.csv       # 清洗+分词+去停用词后的数据
│   ├── test_processed.csv
│   ├── label_mapping.csv         # id ↔ name 映射
│   ├── train_fasttext.txt        # FastText __label__X 格式
│   └── test_fasttext.txt
├── runs/                         # 训练产出 (每次训练独立 timestamp 目录)
│   ├── rf/2026-06-17_XX-XX-XX/   # 随机森林
│   ├── fasttext/2026-06-17_XX-XX-XX/  # FastText
│   └── bert/2026-06-17_XX-XX-XX/     # BERT+LoRA
├── scripts/draw_dag.py           # 模块依赖图生成
└── src/
    ├── preprocess.py             # ★ 数据预处理入口
    ├── models/                   # ★ 三种模型方案
    │   ├── rf/                   # 随机森林: 特征工程 + 模型 + 训练
    │   │   ├── feature_eng.py    #   TF-IDF 向量化 (5,000 维)
    │   │   ├── model.py          #   超参搜索 + 训练 + 存取
    │   │   └── train.py          #   六阶段编排入口
    │   ├── fasttext/             # FastText: n-gram 特征 + 训练
    │   │   ├── feature_eng.py    #   n-gram 参数说明 + 格式转换
    │   │   ├── model.py          #   训练 + 存取 + NumPy2.x 兼容
    │   │   └── train.py          #   六阶段编排入口
    │   └── bert/                 # BERT: Tokenization + LoRA 微调
    │       ├── feature_eng.py    #   BERT Tokenizer + Dataset
    │       ├── model.py          #   预训练加载 → LoRA 注入 → 训练 → merge
    │       └── train.py          #   七阶段编排入口
    └── utils/                    # ★ 三个模型共享的工具
        ├── data_utils.py         #   数据加载 / 清洗 / 分词 / 停用词
        ├── eval_utils.py         #   指标计算 / JSON+MD 持久化
        └── viz_utils.py          #   图表绘制 / 中文字体
```

**设计原则:**
- 每个模型的 `feature_eng` / `model` / `train` 同目录，内聚清晰
- `utils/` 三个模型共用，评估口径 100% 一致
- `config.py` 全局唯一参数来源，零硬编码
- `runs/<model>/<ts>/` 每次训练完整独立，可复现可回溯

---

## 二、完整数据执行流

### Step 0: 预处理 (前置，跑一次)

```
python src/preprocess.py
```

```
data/
  ├── train.csv  (7,273 行 × 3 列: text, label_class, label_id)
  ├── test.csv   (809 行)
  ├── label.txt  (13 类名称)
  └── stopwords.txt (734 词)

        │  src/utils/data_utils.py
        │  preprocess_pipeline() 三个子步骤:
        │
        ├─ ① clean_text()      正则去 HTML/URL/特殊字符, 压缩空白
        ├─ ② segment_text()    jieba.cut() 全模式分词
        └─ ③ filter_tokens()   去掉 734 停用词 + 短词过滤
        │
        ▼
cache/
  ├── train_processed.csv     (7,273 行, 含 text + tokenized_text + label_id)
  ├── test_processed.csv      (809 行)
  ├── label_mapping.csv       (13 行: 0→定义, 1→病因, ...)
  ├── train_fasttext.txt      (__label__治疗方法 肾结石 结石 ...)
  └── test_fasttext.txt
```

---

### 模型训练: 三种模型统一流程

每个模型都遵循相同的六阶段模式，config.py 统一参数入口，utils/ 统一评估+可视化。

---

## 三、模型一: TF-IDF + 随机森林

### 原理

```
文本 → jieba 分词 → TF-IDF 向量化 (5,000 维) → RandomForestClassifier
                                                      │
                                    200 棵决策树并行投票 → 13 类概率 → argmax
```

### 完整训练流程

```
python src/models/rf/train.py
```

```
阶段 ① 特征工程
  cache/train_processed.csv → load_processed_data()
    X_train = list[str] 7,273 条分词文本
    y_train = ndarray, 7,273 个 label_id (0~12)
    X_test  = list[str] 809 条
    label_names = ["定义", "病因", ...] 13 类

  调用: rf/feature_eng.build_tfidf_vectorizer(X_train)
    TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.85)
    → X_train: (7,273, 5,000) scipy sparse matrix
    → X_test:  (809, 5,000)

  调用模块: data_utils, rf/feature_eng
  配置参数: TFIDF_MAX_FEATURES=5000, TFIDF_NGRAM_RANGE=(1,2)

────────────────────────────────────────────────────────────

阶段 ② 超参搜索
  调用: rf/model.search_best_params(X_train, y_train, n_classes=13)

  RandomizedSearchCV(refit=False)
  ├── 参数分布:
  │     n_estimators:      [100,150,200,250,300,400,500]  (7个)
  │     max_depth:         [None,10,15,20,30,40,50]       (7个)
  │     min_samples_split: [2,3,5,7,10]                   (5个)
  │     min_samples_leaf:  [1,2,3,4]                      (4个)
  │     max_features:      ["sqrt","log2",None]           (3个)
  ├── 随机采样 40 组 × StratifiedKFold(cv=4)
  ├── 160 次 fit, f1_weighted 评分
  └── → best_params = {'n_estimators': 200, 'max_depth': None, ...}
      → best_cv_score = 0.5506

  ⚠️ refit=False — 只打分，不最终训练

  输出: runs/rf/<ts>/search_log.csv  (40 组参数 CV 明细)

  调用模块: rf/model
  配置参数: RF_RANDOM_N_ITER=40, RF_RANDOM_CV=4, CV_SCORING=f1_weighted

────────────────────────────────────────────────────────────

阶段 ③ 最终训练
  调用: rf/model.train_final_model(X_train, y_train, best_params)

  RandomForestClassifier(**best_params, n_jobs=-1, class_weight='balanced_subsample')
    .fit(X_train_all, y_train_all)  ← 完整 7,273 条，不拆 CV
    → model = 200 棵树充分生长的随机森林

  调用模块: rf/model
  配置参数: RANDOM_SEED=42, RF_N_JOBS=-1

────────────────────────────────────────────────────────────

阶段 ④ 评估
  调用: eval_utils.evaluate_and_report(y_train, y_train_pred, y_test, y_test_pred)

  compute_metrics() — 统一口径，三个模型共用:
    → 训练: acc=0.9781, precision=0.9788, recall=0.9781, f1=0.9783
    → 测试: acc=0.5847, precision=0.5830, recall=0.5847, f1=0.5817

  持久化:
    save_metrics_json() → runs/rf/<ts>/metrics.json
    save_report_md()    → runs/rf/<ts>/report.md

  调用模块: eval_utils
  配置参数: (使用 sklearn classification_report 默认参数)

────────────────────────────────────────────────────────────

阶段 ⑤ 可视化
  调用: viz_utils.plot_confusion_matrix(y_test_true, y_test_pred, ...)
  调用: viz_utils.plot_feature_importance(model, vectorizer, ...)
  调用: viz_utils.plot_class_distribution(y_train, ...)

  输出:
    runs/rf/<ts>/confusion_matrix.png       (左: 计数, 右: 行归一化)
    runs/rf/<ts>/feature_importance.png     (Top-30 TF-IDF 词)
    runs/rf/<ts>/class_distribution.png     (13 类柱状图)

  调用模块: viz_utils
  配置参数: FIG_DPI=150, FIG_TOP_N=30

────────────────────────────────────────────────────────────

阶段 ⑥ 持久化
  调用: rf/model.save_model(model, vectorizer, run_dir)

  pickle.dump() → runs/rf/<ts>/model.pkl         (约 10 MB)
                 → runs/rf/<ts>/vectorizer.pkl    (TF-IDF 词汇表)
  json.dump()   → runs/rf/<ts>/best_params.json   (超参 + 元信息)

  加载: rf/model.load_model(run_dir) → (model, vectorizer, meta)
```

### 结果

| 指标 | 训练集 | 测试集 | 差值 |
|------|--------|--------|------|
| Accuracy | 0.9781 | 0.5847 | **+39 pp 过拟合** |
| F1 (weighted) | 0.9783 | 0.5817 | +40 pp |

---

## 四、模型二: FastText (n-gram 子词特征)

### 原理

```
文本 → jieba 分词 → __label__X 分词 格式 → fasttext.train_supervised()
                                                 │
                      特征由 fasttext 内部自动完成:
                      ├── 字符 n-gram (minn=2, maxn=3)
                      │     "肾结石" → "<肾","肾结","结石","石>","<肾结","肾结石","结石>"
                      │     效果: 捕获中文子词信息，OOV 也可通过子词组合理解
                      ├── 词级 n-gram (wordNgrams=2)
                      │     "肾结石 怎么 治" → ["肾结石 怎么","怎么 治"]
                      │     效果: 捕获医学短语搭配
                      └── Hierarchical Softmax + 100 维词向量
                           bucket=100,000 哈希桶 (→ 模型 3 MB)
```

### 完整训练流程

```
python src/models/fasttext/train.py

阶段 ① n-gram 特征工程 + 格式转换
  cache/train_fasttext.txt 已存在 → 直接读
  describe_ngram_params() → 打印当前 n-gram 配置
  analyze_ngram_effect()  → 统计: avg_tokens=12.8, median=11

  调用模块: data_utils, fasttext/feature_eng
  配置参数: FASTTEXT_MIN_N=2, FASTTEXT_MAX_N=3, FASTTEXT_WORD_NGRAMS=2

阶段 ② 模型训练
  fasttext.train_supervised(
      input=train_fasttext.txt,   # __label__X 格式
      lr=0.5, epochs=25, dim=50,
      wordNgrams=2, minn=2, maxn=3,
      bucket=100000, loss="softmax",
  )
  → ~2 秒完成
  → 内置评估: test P@1=0.5686

  调用模块: fasttext/model
  配置参数: 以上全部来自 config.py

阶段 ③ 预测
  predict(model, X_train, name_to_id) → [5,0,1,3,...] (7,273 个 label_id)
  predict(model, X_test,  name_to_id) → [5,0,0,0,...] (809 个)

  NumPy 2.x 兼容: fasttext 内部 np.array(copy=False) 被临时 patch

阶段 ④ 评估 (同 RF，eval_utils 统一)
  → 训练: acc=0.9318, f1=0.9325
  → 测试: acc=0.4722, f1=0.4713
  → JSON / MD 持久化

阶段 ⑤ 可视化 (同 RF，viz_utils 统一)
  → confusion_matrix.png, class_distribution.png

阶段 ⑥ 持久化
  model.save_model() → runs/fasttext/<ts>/model.bin (3 MB)
  json.dump()        → best_params.json
```

### 结果

| 指标 | 训练集 | 测试集 | 差值 |
|------|--------|--------|------|
| Accuracy | 0.9318 | 0.4722 | **+46 pp 过拟合** |
| F1 (weighted) | 0.9325 | 0.4713 | +46 pp |

---

## 五、模型三: BERT + LoRA (低秩适配微调) ★ 最佳

### 原理

```
huggingface.co/bert-base-chinese (HF 镜像下载)
  │  12 layer Transformer encoder
  │  每层: Multi-Head Attention (Q/K/V, 768 维, 12 heads)
  │  hidden_dim=768, vocab=21128
  │
  ▼
BertForSequenceClassification (pretrained)
  │  bert.embeddings     +  bert.encoder (12 layer)
  │  + bert.pooler       +  classifier (768→13)
  │  合计: 102,277,645 参数
  │
  ▼ LoRA 注入 (r=8, alpha=16, target=["query","value"])
  │
  │  每层 Q: W_frozen(768×768) + A(768×8)·B(8×768)
  │  每层 V: W_frozen(768×768) + A(768×8)·B(8×768)
  │  classifier: 9,984 + 13
  │
  │  冻结: 102,277,645 (100%)
  │  ★训练:    304,909 (0.3%)
  │
  │  LoRA 为什么有效？
  │  - 预训练权重保留全部语言学知识 (中文句法/语义/常识)
  │  - ΔW = A·B 只需学习"医学领域偏移" (低秩假设)
  │  - 参数量极少 → 小样本不过拟合
  │
  ▼
merge_and_unload()
  合并: frozen_weight + lora_A @ lora_B → 标准 BertForSequenceClassification
  保存: lora_adapter/ (1.3 MB) + checkpoint/ (391 MB)
```

### 完整训练流程

```
python src/models/bert/train.py

阶段 ① Tokenization
  build_tokenizer() → BertTokenizer (vocab=21,128)
  create_datasets(train_df, test_df, tokenizer, max_len=128)
    → MedicalDataset: __getitem__ 时动态 tokenize
    → DataLoader: batch_size=16, pin_memory=True

  调用模块: data_utils, bert/feature_eng
  配置参数: BERT_MODEL_NAME=bert-base-chinese, BERT_MAX_LENGTH=128, BERT_BATCH_SIZE=16

阶段 ② 加载预训练模型
  build_base_model(n_classes=13)
    BertForSequenceClassification.from_pretrained("bert-base-chinese")
    → 102,277,645 参数  (下载 391 MB，镜像: hf-mirror.com)

  调用模块: bert/model
  配置参数: BERT_MODEL_NAME

阶段 ③ LoRA 注入
  LoraConfig(r=8, alpha=16, target=["query","value"], dropout=0.1)
  get_peft_model(base_model, lora_config)

  24 个 attention head 的 Q/V 矩阵各注入:
    lora_A: 768×8 = 6,144 × 2(Q+V) × 12 层 = 147,456
    lora_B: 8×768 = 6,144 × 2(Q+V) × 12 层 = 147,456
    classifier: 768×13 + 13 = 9,997
    ─────────────────────────────────────────
    ★ 可训练 304,909 (0.3%) / 冻结 102,277,645

  调用模块: bert/model  (peft 库)
  配置参数: LORA_R=8, LORA_ALPHA=16, LORA_DROPOUT=0.1, LORA_TARGET_MODULES=["query","value"]

阶段 ④ LoRA 微调
  train_full(model, train_loader, test_loader, device="cuda")
    optimizer: AdamW(lr=5e-4, weight_decay=0.01)
    scheduler: linear warmup (10%) + linear decay
    early stop: patience=2
    梯度裁剪: max_norm=1.0

    Epoch 1/3  train_loss=1.6264  val_loss=1.1929  val_acc=0.5958
    Epoch 2/3  train_loss=1.0848  val_loss=1.1103  val_acc=0.6156
    Epoch 3/3  train_loss=0.9604  val_loss=1.0370  val_acc=0.6341

    455 batch × 3 epoch × GPU ≈ 6 分钟

  调用模块: bert/model
  配置参数: BERT_LR=5e-4, BERT_EPOCHS=3, BERT_WARMUP_RATIO=0.1, BERT_WEIGHT_DECAY=0.01

阶段 ⑤ 评估 (eval_utils 统一)
  → 训练: acc=0.7011, f1=0.7021
  → 测试: acc=0.6341, f1=0.6303

阶段 ⑥ 可视化
  → confusion_matrix.png, class_distribution.png
  → loss_curve.png, accuracy_curve.png

阶段 ⑦ 保存模型
  peft_model.save_pretrained()      → lora_adapter/  (1.3 MB)
  merge_and_unload().save_pretrained() → checkpoint/ (391 MB)
  tokenizer.save_pretrained()          → vocab + config
  json.dump()                          → best_params.json
```

### 结果

| 指标 | 训练集 | 测试集 | 差值 |
|------|--------|--------|------|
| Accuracy | 0.7011 | 0.6341 | **+7 pp** ✓ |
| F1 (weighted) | 0.7021 | 0.6303 | +7 pp |
| ★ 最佳单类 F1 | 化验方案 0.87 (训练) / 0.87 (测试) | | |

---

## 六、三模型完整对比

### 核心指标

```
                    RandomForest    FastText         BERT+LoRA
                    ────────────    ────────         ─────────
原理                TF-IDF + 树     n-gram + HS      Pretrain + LoRA
特征维度            5,000 维        50 维词向量      768 维
模型参数量          200 trees       100K buckets     304K / 102M
训练时间            ~3 min          ~2 sec           ~6 min (GPU)
磁盘占用            ~10 MB          3 MB             1.3 MB (adapter)

训练集 Accuracy      0.9781 ████████  0.9318 ███████  0.7011 ████
训练集 F1            0.9783 ████████  0.9325 ███████  0.7021 ████

测试集 Accuracy      0.5847 ████      0.4722 ███      0.6341 ████★
测试集 F1            0.5817 ████      0.4713 ███      0.6303 ████★

过拟合差 (Train-Test)  39 pp ✗✗✗      46 pp ✗✗✗       7 pp ✓
```

### 各类别 F1 对比

| 类别 | 训练样本 | RF | FastText | BERT+LoRA |
|------|---------|-----|----------|-----------|
| 化验/体检方案 | 192 | 0.99 | 0.90 | **0.87** |
| 预防 | 190 | 0.98 | 0.90 | **0.81** |
| 治疗时间 | 167 | 1.00 | 0.96 | **0.93** |
| 所属科室 | 116 | 0.99 | 0.96 | **0.97** |
| 传染性 | 143 | 0.98 | 0.97 | **0.93** |
| 临床表现 | 1,031 | 0.97 | 0.94 | 0.53 |
| 病因 | 785 | 0.98 | 0.94 | 0.50 |
| 治疗方法 | 1,632 | 0.99 | 0.93 | 0.68 |
| 定义 | 406 | 0.93 | 0.87 | 0.64 |
| 相关病症 | 251 | 0.97 | 0.88 | 0.43 |
| 治愈率 | 240 | 0.99 | 0.91 | 0.68 |
| 禁忌 | 315 | 0.98 | 0.91 | 0.49 |
| 其他 | 1,805 | 0.98 | 0.96 | 0.63 |

### 为什么 BERT+LoRA 更好？

```
                    RF / FastText                    BERT + LoRA
                    ────────────                     ──────────
文本理解            词袋模型，无顺序                  12 层 Self-Attention 全局建模
                    "不 发烧" ≈ "发烧 不"             "不" 修饰 "发烧" → 否定语义被捕获

OOV 泛化            未见过的词 = 0 贡献               Subword tokenizer + 预训练语义
                    TF-IDF 词汇表外完全盲              "阿莫西林克拉维酸钾" 可拆分子词理解

领域迁移            从零学                              已有中文常识 + 只需学医学偏移
                    7K 样本学 5K 特征 → 过拟合         LoRA 只调 0.3% 参数 → 不过拟合

不均衡处理          class_weight='balanced_subsample'  Cross-entropy + LoRA 天然鲁棒

特征层次            单一 (TF-IDF / n-gram)             多层抽象
                    "头痛 发热 咳嗽" → 三个词           L1: 词级 → L6: 症状组合 → L12: 疾病判别
```

---

## 七、模块依赖 DAG

```
                    ┌──────────────┐
                    │  config.py   │ ← 全局参数, 所有模块只读
                    └──┬──┬──┬──┬──┘
         ┌─────────────┘  │  │  └─────────────┐
         ▼                ▼  ▼                ▼
   ┌──────────┐    ┌────────────┐    ┌──────────────┐
   │data_utils│    │feature_eng │    │  eval_utils  │ ← 三个模型通用
   │ 加载/清洗 │    │ TF-IDF/n-  │    │ 指标/JSON/MD  │
   └────┬─────┘    │ gram/token │    └──────┬───────┘
        │          └─────┬──────┘           │
        │                │                  │
        ▼                ▼                  ▼
  ┌─────────────────────────────────────────────┐
  │              train.py (编排层)               │
  │  ① 加载 ② 特征 ③ 训练 ④ 评估 ⑤ 可视化 ⑥ 保存 │
  │              rf / fasttext / bert             │
  └─────────────────────┬───────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  runs/<model>/   │
              │  <timestamp>/    │ ← 每次训练完整产出
              └──────────────────┘

  单向无环:
    - train.py 只调用各模块，不反向依赖
    - 模块之间零耦合 (data_utils 不知道 feature_eng 存在)
    - 全部依赖 config.py (只读, 不写回)
```

---

## 八、使用指南

```bash
# 1. 预处理 (一次)
python src/preprocess.py

# 2. 随机森林 (~3 min)
python src/models/rf/train.py

# 3. FastText (~2 sec)
python src/models/fasttext/train.py

# 4. BERT + LoRA (~6 min, 需 GPU)
python src/models/bert/train.py

# 5. 仅评估 (基于已保存模型, 不重新训练)
python src/models/rf/train.py       --eval-only
python src/models/fasttext/train.py --eval-only
python src/models/bert/train.py     --eval-only
```

---

## 九、结论

| 维度 | 结论 |
|------|------|
| **最佳模型** | BERT + LoRA (r=8) — 测试 F1 0.6303, 过拟合仅 7pp |
| **最快速** | FastText — 2 秒训练, 但严重过拟合 |
| **可解释性** | RF — 特征重要性 Top-30 可直观理解 |
| **可改进** | 数据增强 (对"其他" 类采样), 增大 LoRA r=16, 更多 epoch |
| **工程亮点** | 统一 config + 统一 eval + 独立 runs 目录 + DAG 架构 |
