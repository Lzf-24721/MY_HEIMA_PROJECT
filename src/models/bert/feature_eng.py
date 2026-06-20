"""
==============================================================================
  BERT 特征工程 — Tokenization + Dataset
  medical_classify/src/models/bert/feature_eng.py

  对外 API:
    build_tokenizer(model_name) → BertTokenizer
    create_datasets(train_df, test_df, tokenizer, max_len) → (train_ds, test_ds)
==============================================================================
"""

import os
import sys
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config as cfg

# 国内 Hugging Face 镜像（网络不可达时自动 fallback）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def build_tokenizer(model_name=None):
    """加载 BERT 中文分词器"""
    from transformers import BertTokenizer
    if model_name is None:
        model_name = cfg.BERT_MODEL_NAME
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"[INFO] BERT Tokenizer: {model_name}  (vocab={tokenizer.vocab_size})")
    return tokenizer


class _MedicalDataset(Dataset):
    """PyTorch Dataset: 每次 __getitem__ 做 tokenization（不预缓存，省内存）"""
    def __init__(self, df, tokenizer, max_len):
        self.texts   = df["tokenized_text"].fillna("").astype(str).tolist()
        self.labels  = df["label_id"].values.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_datasets(train_df, test_df, tokenizer, max_length=None):
    """DataFrame → PyTorch Dataset"""
    if max_length is None:
        max_length = cfg.BERT_MAX_LENGTH
    train_ds = _MedicalDataset(train_df, tokenizer, max_length)
    test_ds  = _MedicalDataset(test_df,  tokenizer, max_length)
    print(f"[INFO] BERT Dataset: 训练 {len(train_ds)} 条, 测试 {len(test_ds)} 条  "
          f"(max_len={max_length})")
    return train_ds, test_ds
