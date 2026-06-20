# 医学文本分类 — 模型评估报告

- 模型: FastText
- 运行ID: 2026-06-17_19-16-41
- 生成时间: 2026-06-17 19:17:23
- 训练样本: 7273
- 测试样本: 809
- 特征维度: 50
- 类别数: 13

## 超参搜索
- CV 评分: `N/A (no CV search)`
- 最佳超参: `{'lr': 0.5, 'epochs': 25, 'dim': 50, 'wordNgrams': 2, 'minn': 2, 'maxn': 3}`

## 训练集
| Metric | Value |
|--------|-------|
| Accuracy | 0.9318 |
| Precision (weighted) | 0.9374 |
| Recall (weighted) | 0.9318 |
| F1 (weighted) | 0.9325 |

## 测试集
| Metric | Value |
|--------|-------|
| Accuracy | 0.4722 |
| Precision (weighted) | 0.4833 |
| Recall (weighted) | 0.4722 |
| F1 (weighted) | 0.4713 |