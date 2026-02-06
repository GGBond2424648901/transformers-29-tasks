# 🎯 零样本文本分类（Zero-shot Text Classification）

## 📖 任务简介

零样本分类是一种无需训练数据即可对文本进行分类的技术。你只需要提供候选标签，模型就能判断文本属于哪个类别，非常适合快速原型开发和灵活的分类需求。

## 🎯 应用场景

- **📧 邮件分类**: 自动分类收件箱、垃圾邮件过滤
- **🛍️ 电商分类**: 商品自动分类、评论情感分析
- **📰 内容审核**: 新闻分类、敏感内容检测
- **💬 客服系统**: 问题分类、意图识别
- **🔍 信息检索**: 文档分类、主题聚类

## 📁 文件说明

- `零样本分类示例.py` - Pipeline 推理示例（推荐入门）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install transformers torch
```

### 2. 运行示例

```bash
python 零样本分类示例.py
```

## 💡 使用示例

### 基础分类

```python
from transformers import pipeline

# 创建分类器
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# 分类文本
text = "这部电影真的太精彩了！"
candidate_labels = ["正面评价", "负面评价", "中性评价"]

result = classifier(text, candidate_labels)

print(result['labels'][0])  # 最可能的标签
print(result['scores'][0])  # 置信度
```

### 多标签分类

```python
# 允许多个标签同时为真
result = classifier(
    text,
    candidate_labels,
    multi_label=True
)

# 筛选置信度高的标签
for label, score in zip(result['labels'], result['scores']):
    if score > 0.5:
        print(f"{label}: {score:.2%}")
```

### 自定义假设模板

```python
# 使用自定义模板提高准确性
result = classifier(
    text,
    candidate_labels,
    hypothesis_template="这段文本是关于{}的。"
)
```

## 🎨 推荐模型

### 英文模型

| 模型 | 大小 | 特点 |
|------|------|------|
| facebook/bart-large-mnli | 大 | 效果最好，推荐使用 |
| typeform/distilbert-base-uncased-mnli | 小 | 速度快，适合实时应用 |
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 中 | 平衡性能和速度 |

### 中文模型

| 模型 | 特点 |
|------|------|
| uer/roberta-base-finetuned-chinanews-chinese | 中文新闻分类 |
| IDEA-CCNL/Erlangshen-Roberta-110M-NLI | 中文自然语言推理 |

## 📊 输出说明

零样本分类的输出包含：
- `labels`: 按置信度排序的标签列表
- `scores`: 对应的置信度分数（0-1）
- `sequence`: 输入的文本

## 💡 使用技巧

### 1. 标签设计

✅ **好的标签**:
- 清晰、具体
- 互相独立
- 可以使用短语或句子

```python
# 好的标签
labels = ["正面评价", "负面评价", "中性评价"]

# 更好的标签（更具体）
labels = ["非常满意", "比较满意", "一般", "不满意", "非常不满意"]
```

❌ **不好的标签**:
- 模糊、重叠
- 过于宽泛

```python
# 不好的标签
labels = ["好", "不好", "还行"]  # 太模糊
```

### 2. 性能优化

```python
# 1. 减少候选标签数量
labels = ["科技", "体育", "娱乐"]  # 而不是 10+ 个标签

# 2. 批量处理
texts = ["文本1", "文本2", "文本3"]
results = classifier(texts, labels)

# 3. 使用更小的模型
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)
```

### 3. 提高准确性

```python
# 1. 使用假设模板
result = classifier(
    text,
    labels,
    hypothesis_template="这是一条关于{}的新闻。"
)

# 2. 多标签分类
result = classifier(
    text,
    labels,
    multi_label=True
)

# 3. 设置阈值
threshold = 0.5
predicted_labels = [
    label for label, score in zip(result['labels'], result['scores'])
    if score > threshold
]
```

## 🎯 实际应用示例

### 1. 情感分析

```python
text = "这个产品质量很好，值得购买！"
labels = ["正面", "负面", "中性"]
result = classifier(text, labels)
```

### 2. 意图识别

```python
query = "我想订一张去北京的机票"
intents = ["订票", "查询天气", "设置提醒", "推荐服务"]
result = classifier(query, intents)
```

### 3. 新闻分类

```python
news = "科技公司发布了最新的人工智能模型"
categories = ["科技", "体育", "娱乐", "财经", "政治"]
result = classifier(news, categories)
```

### 4. 商品特征提取

```python
description = "这款手机配备了强大的摄像头和长续航电池"
features = ["摄像功能", "电池续航", "价格优势", "屏幕显示"]
result = classifier(description, features, multi_label=True)
```

## ⚠️ 注意事项

1. **模型语言**: 确保模型支持你的文本语言
2. **标签数量**: 标签过多会影响准确性和速度
3. **文本长度**: 过长的文本可能需要截断
4. **置信度**: 低置信度结果需要谨慎使用

## 🆚 与传统分类的对比

| 特性 | 零样本分类 | 传统分类 |
|------|-----------|---------|
| 训练数据 | ❌ 不需要 | ✅ 需要大量标注数据 |
| 灵活性 | ✅ 可随时更改类别 | ❌ 需要重新训练 |
| 准确性 | 中等 | 高（针对特定任务） |
| 速度 | 较快 | 快 |
| 适用场景 | 快速原型、灵活分类 | 固定类别、高精度需求 |

## 🔗 相关资源

- [Zero-shot Learning 介绍](https://huggingface.co/tasks/zero-shot-classification)
- [BART 论文](https://arxiv.org/abs/1910.13461)
- [自然语言推理（NLI）介绍](https://nlp.stanford.edu/projects/snli/)
