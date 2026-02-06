#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码词填充（Fill-Mask）实战示例
使用 BERT 等模型进行掩码词预测
"""

import os

# 设置模型缓存路径
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline

print("=" * 70)
print("🎭 掩码词填充实战示例")
print("=" * 70)
print(f"📁 模型缓存路径: {os.environ['HF_HOME']}")
print("=" * 70)

# 1. 创建掩码词填充 pipeline
print("\n📦 步骤 1: 加载模型")
print("-" * 70)

# 使用中文 BERT 模型
unmasker = pipeline(
    "fill-mask",
    model="bert-base-chinese"
)

print("✅ 模型加载成功！")
print(f"   模型: bert-base-chinese")
print(f"   任务: 掩码词填充")

# 2. 基础填充示例
print("\n🔍 步骤 2: 基础掩码词填充")
print("-" * 70)

text = "今天天气真[MASK]！"
print(f"输入文本: {text}")

results = unmasker(text)

print("\n✅ 填充完成！")
print(f"\n📊 预测结果（Top 5）:")
for i, result in enumerate(results, 1):
    print(f"   {i}. {result['token_str']:<10} 置信度: {result['score']:.2%}")
    print(f"      完整句子: {result['sequence']}")

# 3. 多个掩码词
print("\n" + "=" * 70)
print("📝 步骤 3: 多个掩码词填充")
print("=" * 70)

sentences = [
    "我喜欢吃[MASK]。",
    "他是一位[MASK]的科学家。",
    "这本书非常[MASK]。",
    "北京是中国的[MASK]。"
]

for sentence in sentences:
    print(f"\n输入: {sentence}")
    results = unmasker(sentence, top_k=3)
    
    print("预测:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['token_str']:<8} ({result['score']:.2%}) - {result['sequence']}")

# 4. 句子补全
print("\n" + "=" * 70)
print("✍️  步骤 4: 句子补全")
print("=" * 70)

incomplete_sentences = [
    "人工智能是[MASK]的未来。",
    "学习编程需要[MASK]和耐心。",
    "健康的生活方式包括[MASK]和运动。"
]

print("句子补全示例：\n")

for sentence in incomplete_sentences:
    results = unmasker(sentence, top_k=1)
    completed = results[0]['sequence']
    word = results[0]['token_str']
    
    print(f"原句: {sentence}")
    print(f"补全: {completed}")
    print(f"填入词: {word} (置信度: {results[0]['score']:.2%})\n")

# 5. 文本纠错
print("=" * 70)
print("🔧 步骤 5: 文本纠错（实验性）")
print("=" * 70)

# 将可能错误的词替换为 [MASK]，让模型预测正确的词
error_texts = [
    ("我[MASK]天去了公园。", "昨"),  # 正确应该是"昨"
    ("这个问题很[MASK]单。", "简"),  # 正确应该是"简"
]

print("文本纠错示例：\n")

for text, expected in error_texts:
    results = unmasker(text, top_k=3)
    
    print(f"输入: {text}")
    print(f"期望: {expected}")
    print("预测:")
    for i, result in enumerate(results, 1):
        is_correct = "✅" if result['token_str'] == expected else "  "
        print(f"   {is_correct} {i}. {result['token_str']:<8} ({result['score']:.2%})")
    print()

# 6. 使用技巧
print("=" * 70)
print("💡 使用技巧")
print("=" * 70)
print("""
掩码词填充的使用技巧：

1. 掩码标记：
   - BERT 中文: [MASK]
   - BERT 英文: [MASK]
   - RoBERTa: <mask>
   - 不同模型使用不同的掩码标记

2. 控制输出数量：
   results = unmasker(text, top_k=10)  # 返回前 10 个预测

3. 多个掩码：
   - 一次只能填充一个 [MASK]
   - 多个掩码需要分别处理

4. 上下文很重要：
   - 提供足够的上下文信息
   - 上下文越丰富，预测越准确

示例代码：

# 指定返回数量
results = unmasker("今天天气真[MASK]！", top_k=10)

# 获取最佳预测
best_prediction = results[0]['token_str']

# 获取完整句子
completed_sentence = results[0]['sequence']
""")

# 7. 应用场景
print("\n" + "=" * 70)
print("🎯 应用场景")
print("=" * 70)
print("""
掩码词填充的主要应用：

1. 📝 智能输入法
   - 词语联想
   - 自动补全
   - 输入建议

2. 🔧 文本纠错
   - 拼写检查
   - 语法纠正
   - 错别字修正

3. 📚 语言学习
   - 填空练习
   - 词汇测试
   - 语境理解

4. 🤖 对话系统
   - 句子补全
   - 意图理解
   - 上下文推理

5. 📊 数据增强
   - 生成相似句子
   - 扩充训练数据
   - 同义词替换
""")

# 8. 模型推荐
print("\n" + "=" * 70)
print("🎨 模型推荐")
print("=" * 70)
print("""
中文模型：
- bert-base-chinese: 通用中文 BERT
- hfl/chinese-roberta-wwm-ext: 中文 RoBERTa（效果更好）
- hfl/chinese-bert-wwm-ext: 中文 BERT WWM

英文模型：
- bert-base-uncased: 通用英文 BERT
- roberta-base: 英文 RoBERTa
- albert-base-v2: 轻量级 ALBERT

使用方法：
unmasker = pipeline("fill-mask", model="hfl/chinese-roberta-wwm-ext")
""")

# 9. 性能对比
print("\n" + "=" * 70)
print("⚡ 不同模型性能对比")
print("=" * 70)

test_sentence = "我喜欢[MASK]编程。"

models = [
    "bert-base-chinese",
    # "hfl/chinese-roberta-wwm-ext",  # 取消注释以测试
]

print(f"测试句子: {test_sentence}\n")

for model_name in models:
    print(f"模型: {model_name}")
    try:
        temp_unmasker = pipeline("fill-mask", model=model_name)
        results = temp_unmasker(test_sentence, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['token_str']:<8} ({result['score']:.2%})")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
    print()

print("=" * 70)
print("✨ 示例完成！")
print("=" * 70)
