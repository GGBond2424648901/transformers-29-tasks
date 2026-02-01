#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型进行情感分析预测
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

print("=" * 60)
print("加载训练好的模型...")
print("=" * 60)

# 获取脚本所在目录的绝对路径
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "my_sentiment_model")

print(f"模型路径: {model_path}")

# 检查模型是否存在
if not os.path.exists(model_path):
    print(f"\n❌ 错误：找不到模型文件夹")
    print(f"   期望位置: {model_path}")
    print(f"   当前目录: {os.getcwd()}")
    print(f"\n💡 请先运行 Trainer_实战示例.py 训练模型")
    exit(1)

# 1. 加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"✓ 模型加载成功！")
print(f"✓ 模型类型: {model.config.model_type}")
print(f"✓ 参数量: {sum(p.numel() for p in model.parameters()):,}")
print()

# 2. 准备测试文本
test_texts = [
    "这个产品质量很好，非常满意！",
    "太差了，完全不值这个价格",
    "还可以，一般般",
    "物流很快，包装完好，推荐购买",
    "客服态度恶劣，再也不买了",
    "性价比不错，值得入手",
]

print("=" * 60)
print("开始预测...")
print("=" * 60)

# 3. 对每个文本进行预测
for i, text in enumerate(test_texts, 1):
    # 分词
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # 预测（不计算梯度，节省内存）
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 计算概率
    probs = F.softmax(outputs.logits, dim=-1)
    negative_prob = probs[0][0].item()
    positive_prob = probs[0][1].item()
    
    # 判断情感
    sentiment = "正面 😊" if positive_prob > negative_prob else "负面 😞"
    confidence = max(positive_prob, negative_prob)
    
    # 输出结果
    print(f"\n{i}. 文本: {text}")
    print(f"   预测: {sentiment}")
    print(f"   置信度: {confidence:.2%}")
    print(f"   详细: 正面={positive_prob:.2%}, 负面={negative_prob:.2%}")

print("\n" + "=" * 60)
print("预测完成！")
print("=" * 60)

# 4. 交互式预测
print("\n" + "=" * 60)
print("交互式预测（输入 'q' 退出）")
print("=" * 60)

while True:
    user_input = input("\n请输入要分析的文本: ").strip()
    
    if user_input.lower() == 'q':
        print("再见！")
        break
    
    if not user_input:
        print("⚠️  请输入有效文本")
        continue
    
    # 预测
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1)
    negative_prob = probs[0][0].item()
    positive_prob = probs[0][1].item()
    
    sentiment = "正面 😊" if positive_prob > negative_prob else "负面 😞"
    confidence = max(positive_prob, negative_prob)
    
    print(f"\n   预测结果: {sentiment}")
    print(f"   置信度: {confidence:.2%}")
    print(f"   详细概率: 正面={positive_prob:.2%}, 负面={negative_prob:.2%}")
