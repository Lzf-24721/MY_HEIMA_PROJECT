"""
==============================================================================
  BERT + LoRA — 预训练模型注入低秩适配器，微调分类头
  medical_classify/src/models/bert/model.py

  LoRA 原理:
    - 冻结 BERT 全部 102M 参数，只训练注入的 LoRA 矩阵 + 分类头
    - 每层 attention Q/V 权重 W (768×768) 拆成 W + ΔW，ΔW = A·B
    - A: 768×r, B: r×768  (r=8 → 仅 12K 参数替代 590K)
    - 24 层 × 12K + 分类头 ≈ 300K 可训练参数 (0.3% of 102M)

  流程:
    ① build_base_model()       → 下载/加载 bert-base-chinese
    ② apply_lora(model)        → 注入 LoRA adapters → PEFT model
    ③ train_full(...)           → 只训练 LoRA weights + classifier
    ④ merge_model(model)       → 合并 LoRA 回原权重 → 标准 BERT (保存用)
    ⑤ save / load

  对外 API:
    build_base_model(n_classes)              → BertForSequenceClassification (裸)
    apply_lora(base_model)                   → PeftModel
    inspect_trainable(model)                 → 打印可训练参数分布
    train_full(model, ...)                   → history
    merge_model(peft_model)                  → BertForSequenceClassification (已合并)
    save_lora_model(model, tokenizer, run_dir) → 保存 adapter + merged
    load_lora_model(run_dir, n_classes)      → (merged_model, tokenizer, meta)
==============================================================================
"""

import os, sys, json
from datetime import datetime
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import config as cfg

import torch
from torch.optim import AdamW


# ==================== ① 基础模型 ==========================

def build_base_model(n_classes):
    """
    下载/加载 bert-base-chinese + 分类头 (全量 102M 参数)。

    ⚠️ 此时所有参数 requires_grad=True，下一步 apply_lora() 会冻结。
    """
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained(
        cfg.BERT_MODEL_NAME,
        num_labels=n_classes,
        hidden_dropout_prob=cfg.BERT_DROPOUT,
    )
    print(f"[INFO] 基础模型已加载: {cfg.BERT_MODEL_NAME} "
          f"(n_classes={n_classes}, total={sum(p.numel() for p in model.parameters()):,})")
    return model


# ==================== ② LoRA 注入 ==========================

def apply_lora(base_model):
    """
    在 BERT 的 Q/V 注意力矩阵上注入 LoRA adapters。
    注入后冻结所有原始参数，只留 LoRA + classifier 可训练。

    返回: PeftModel (带 LoRA adapters)
    """
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


# ==================== ③ 训练循环 ==========================

def compute_class_weights(y_train, n_classes):
    """从训练标签计算类别权重: w_i = total / (n_classes * count_i)"""
    counts = np.bincount(y_train, minlength=n_classes)
    weights = len(y_train) / (n_classes * counts)
    weights = np.clip(weights, 0.5, 10.0)  # 限制极端权重
    return torch.tensor(weights, dtype=torch.float)


_loss_fn_cache = None

def _weighted_loss(logits, labels, class_weights=None):
    """类别加权 CrossEntropyLoss"""
    global _loss_fn_cache
    if class_weights is not None:
        device = logits.device
        if _loss_fn_cache is None or not torch.equal(_loss_fn_cache.get('weights', torch.tensor([])).to(device), class_weights.to(device)):
            _loss_fn_cache = {'fn': torch.nn.CrossEntropyLoss(weight=class_weights.to(device)), 'weights': class_weights}
    else:
        _loss_fn_cache = {'fn': torch.nn.CrossEntropyLoss(), 'weights': None}
    return _loss_fn_cache['fn'](logits, labels)


def _train_one_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if class_weights is not None:
            loss = _weighted_loss(logits, labels, class_weights.to(device))
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.BERT_GRADIENT_CLIP)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def _validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(dataloader), correct / total


def train_full(model, train_loader, val_loader, device="cuda",
               epochs=None, lr=None, class_weights=None):
    """LoRA 训练循环 (warmup + 早停 + class weights)"""
    from transformers import get_linear_schedule_with_warmup

    if epochs is None:
        epochs = cfg.BERT_EPOCHS
    if lr is None:
        lr = cfg.BERT_LR
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    model = model.to(device)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg.BERT_WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.BERT_WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
        val_loss, val_acc = _validate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.BERT_EARLY_STOP:
                print(f"  Early stop at epoch {epoch} (best: epoch {best_epoch})")
                break

    return history


# ==================== ④ 合并 ===============================

def merge_model(peft_model):
    """
    将 LoRA adapter 与 frozen base weights 合并 → 标准 BertForSequenceClassification。
    合并后可直接 pickle 或 save_pretrained，无需 peft 库加载。
    """
    merged = peft_model.merge_and_unload()
    total = sum(p.numel() for p in merged.parameters())
    print(f"[INFO] LoRA 已合并到基础模型 ({total:,} params)")
    return merged


# ==================== ⑤ 存取 ===============================

def save_lora_model(peft_model, tokenizer, run_dir, extra_meta=None):
    """保存 LoRA adapter + 合并模型"""
    run_dir = cfg.Path(run_dir) if not isinstance(run_dir, cfg.Path) else run_dir
    adapter_dir = str(run_dir / "lora_adapter")
    merged_dir  = str(run_dir / "checkpoint")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(merged_dir,  exist_ok=True)

    # A) 保存 LoRA adapter (轻量，可继续训练)
    peft_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    size_mb = sum(os.path.getsize(os.path.join(adapter_dir, f))
                  for f in os.listdir(adapter_dir) if os.path.isfile(os.path.join(adapter_dir, f))) / (1024*1024)
    print(f"[INFO] LoRA adapter 已保存: {adapter_dir} ({size_mb:.1f} MB)")

    # B) 合并并保存完整模型 (eval 直接可用)
    merged = merge_model(peft_model)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[INFO] 合并模型已保存: {merged_dir}")

    # 元信息
    meta = {
        "saved_at": datetime.now().isoformat(),
        "model_type": "BERT + LoRA",
        "base_model": cfg.BERT_MODEL_NAME,
        "lora_r": cfg.LORA_R,
        "lora_alpha": cfg.LORA_ALPHA,
        "lora_target_modules": cfg.LORA_TARGET_MODULES,
        "max_length": cfg.BERT_MAX_LENGTH,
    }
    if extra_meta:
        meta.update(extra_meta)

    json_path = str(run_dir / cfg.RUN_BEST_PARAMS_JSON)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 元信息已保存: {json_path}")


def load_lora_model(run_dir, n_classes=None):
    """
    从 checkpoint/ 加载合并后的完整模型（eval 时用）。
    从 lora_adapter/ 加载 LoRA adapter（进一步训练用，需 peft）。
    """
    from transformers import BertForSequenceClassification

    run_dir = cfg.Path(run_dir) if not isinstance(run_dir, cfg.Path) else run_dir
    merged_dir = str(run_dir / "checkpoint")

    if not os.path.exists(merged_dir):
        raise FileNotFoundError(f"模型不存在: {merged_dir}")

    if n_classes is None:
        with open(os.path.join(merged_dir, "config.json"), "r") as f:
            n_classes = json.load(f).get("num_labels", 13)

    model = BertForSequenceClassification.from_pretrained(merged_dir)
    tokenizer = None
    try:
        from src.models.bert.feature_eng import build_tokenizer
        tokenizer = build_tokenizer(cfg.BERT_MODEL_NAME)
    except Exception:
        pass

    print(f"[INFO] 模型已加载: {merged_dir} ({n_classes} 类)")

    meta = {}
    json_path = str(run_dir / cfg.RUN_BEST_PARAMS_JSON)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, tokenizer, meta


def inspect_trainable(model):
    """打印可训练参数分布（诊断用）"""
    lines = ["[INFO] 可训练参数分布:"]
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            lines.append(f"  {name:55s} {param.numel():>10,}")
            total_trainable += param.numel()
    lines.append(f"  {'TOTAL':55s} {total_trainable:>10,}")
    print("\n".join(lines))
