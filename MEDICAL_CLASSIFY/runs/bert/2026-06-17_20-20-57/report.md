# 医学文本分类 — 模型评估报告

- 模型: BERT + LoRA (bert-base-chinese)
- 运行ID: 2026-06-17_20-20-57
- 生成时间: 2026-06-17 20:24:58
- 训练样本: 7273
- 测试样本: 809
- 特征维度: 768
- 类别数: 13

## 超参搜索
- CV 评分: `N/A`
- 最佳超参: `{'lr': 0.0005, 'epochs': 3, 'lora_r': 8, 'lora_alpha': 16, 'lora_target': ['query', 'value']}`

## 训练集
| Metric | Value |
|--------|-------|
| Accuracy | 0.7011 |
| Precision (weighted) | 0.7133 |
| Recall (weighted) | 0.7011 |
| F1 (weighted) | 0.7021 |

## 测试集
| Metric | Value |
|--------|-------|
| Accuracy | 0.6341 |
| Precision (weighted) | 0.6468 |
| Recall (weighted) | 0.6341 |
| F1 (weighted) | 0.6303 |