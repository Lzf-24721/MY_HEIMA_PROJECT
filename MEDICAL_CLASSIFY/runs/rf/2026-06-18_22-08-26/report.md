# 医学文本分类 — 模型评估报告

- 模型: RandomForest
- 运行ID: 2026-06-18_22-08-26
- 生成时间: 2026-06-18 22:15:20
- 训练样本: 7273
- 测试样本: 809
- 特征维度: 5000
- 类别数: 13

## 超参搜索
- CV 评分: `0.5356`
- 最佳超参: `{'n_estimators': 250, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_features': 0.5, 'max_depth': None}`

## 训练集
| Metric | Value |
|--------|-------|
| Accuracy | 0.8461 |
| Precision (weighted) | 0.8578 |
| Recall (weighted) | 0.8461 |
| F1 (weighted) | 0.8465 |

## 测试集
| Metric | Value |
|--------|-------|
| Accuracy | 0.5612 |
| Precision (weighted) | 0.5610 |
| Recall (weighted) | 0.5612 |
| F1 (weighted) | 0.5560 |