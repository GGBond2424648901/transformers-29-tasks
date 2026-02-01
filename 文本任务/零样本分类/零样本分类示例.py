#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零样本文本分类实战示例
无需训练即可对文本进行分类
"""

import os

# 设置模型缓存路径
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline

print("=" * 70)
print("🎯 零样本文本分类实战示例")
print("=" * 70)
print(f"📁 模型缓存路径: {os.environ['HF_HOME']}")
print("=" * 70)

# 1. 创建零样本分类 pipeline
print("\n📦 步骤 1: 加载模型")
print("-" * 70)

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

print("✅ 模型加载成功！")
print(f"   模型: facebook/bart-large-mnli")
print(f"   任务: 零样本分类")

# 2. 基础分类示例
print("\n🔍 步骤 2: 基础分类")
print("-" * 70)

text = "这部电影真的太精彩了，我非常喜欢！"
candidate_labels = ["正面评价", "负面评价", "中性评价"]

print(f"文本: {text}")
print(f"候选标签: {candidate_labels}")

result = classifier(text, candidate_labels)

print("\n✅ 分类完成！")
print(f"\n📊 分类结果:")
for label, score in zip(result['labels'], result['scores']):
    print(f"   {label:<15} 置信度: {score:.2%}")

# 3. 新闻分类示例
print("\n" + "=" * 70)
print("📰 步骤 3: 新闻分类")
print("=" * 70)

news_text = "科技公司发布了最新的人工智能模型，性能提升显著。"
news_labels = ["科技", "体育", "娱乐", "财经", "政治"]

print(f"新闻: {news_text}")
print(f"候选类别: {news_labels}")

result = classifier(news_text, news_labels)

print("\n📊 分类结果:")
for label, score in zip(result['labels'], result['scores']):
    print(f"   {label:<10} 置信度: {score:.2%}")

# 4. 意图识别示例
print("\n" + "=" * 70)
print("💬 步骤 4: 意图识别")
print("=" * 70)

user_queries = [
    "我想订一张去北京的机票",
    "今天天气怎么样？",
    "帮我设置一个明天早上8点的闹钟",
    "推荐一家附近的餐厅"
]

intent_labels = ["订票", "查询天气", "设置提醒", "推荐服务"]

print("用户查询意图识别：\n")

for query in user_queries:
    result = classifier(query, intent_labels)
    top_intent = result['labels'][0]
    top_score = result['scores'][0]
    
    print(f"查询: {query}")
    print(f"意图: {top_intent} (置信度: {top_score:.2%})\n")

# 5. 多标签分类
print("=" * 70)
print("🏷️  步骤 5: 多标签分类")
print("=" * 70)

article = "这款智能手机配备了强大的摄像头和长续航电池，价格也很实惠。"
feature_labels = ["摄像功能", "电池续航", "价格优势", "屏幕显示", "性能配置"]

print(f"商品描述: {article}")
print(f"特征标签: {feature_labels}")

result = classifier(article, feature_labels, multi_label=True)

print("\n📊 特征匹配结果:")
for label, score in zip(result['labels'], result['scores']):
    if score > 0.5:  # 只显示置信度大于 50% 的标签
        print(f"   ✅ {label:<15} 置信度: {score:.2%}")
    else:
        print(f"   ❌ {label:<15} 置信度: {score:.2%}")

# 6. 批量分类
print("\n" + "=" * 70)
print("📦 步骤 6: 批量分类")
print("=" * 70)

texts = [
    "这个产品质量很好，值得购买！",
    "服务态度太差了，非常失望。",
    "价格适中，性价比还可以。"
]

sentiment_labels = ["正面", "负面", "中性"]

print("批量情感分析：\n")

results = classifier(texts, sentiment_labels)

for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"情感: {result['labels'][0]} (置信度: {result['scores'][0]:.2%})\n")

# 7. 使用技巧
print("=" * 70)
print("💡 使用技巧")
print("=" * 70)
print("""
零样本分类的优势和技巧：

1. ✨ 无需训练数据
   - 直接使用预训练模型
   - 快速原型开发
   - 灵活调整类别

2. 🎯 标签设计建议
   - 使用清晰、具体的标签
   - 避免标签之间重叠
   - 可以使用短语或句子

3. 📊 多标签分类
   - 设置 multi_label=True
   - 允许多个标签同时为真
   - 适合特征提取

4. ⚡ 性能优化
   - 减少候选标签数量
   - 使用更小的模型
   - 批量处理文本

示例代码：

# 使用假设模板（hypothesis template）
result = classifier(
    text,
    candidate_labels,
    hypothesis_template="这段文本是关于{}的。"
)

# 多标签分类
result = classifier(
    text,
    candidate_labels,
    multi_label=True
)
""")

# 8. 应用场景
print("\n" + "=" * 70)
print("🎯 应用场景")
print("=" * 70)
print("""
零样本分类的主要应用：

1. 📧 邮件分类
   - 自动分类收件箱
   - 垃圾邮件过滤
   - 优先级排序

2. 🛍️ 电商分类
   - 商品自动分类
   - 评论情感分析
   - 用户意图识别

3. 📰 内容审核
   - 新闻分类
   - 敏感内容检测
   - 主题标签

4. 💬 客服系统
   - 问题分类
   - 意图识别
   - 自动路由

5. 🔍 信息检索
   - 文档分类
   - 相关性判断
   - 主题聚类
""")

# 9. 中文模型推荐
print("\n" + "=" * 70)
print("🇨🇳 中文模型推荐")
print("=" * 70)
print("""
对于中文文本，可以尝试以下模型：

1. uer/roberta-base-finetuned-chinanews-chinese
   - 专门针对中文新闻
   - 分类效果好

2. IDEA-CCNL/Erlangshen-Roberta-110M-NLI
   - 中文自然语言推理
   - 适合零样本分类

使用方法：
classifier = pipeline(
    "zero-shot-classification",
    model="uer/roberta-base-finetuned-chinanews-chinese"
)
""")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
