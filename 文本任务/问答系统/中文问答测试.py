#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文问答系统测试
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline

print("=" * 70)
print("❓ 中文问答系统测试")
print("=" * 70)

# 加载模型
print("\n📦 加载模型...")

# 使用训练好的模型
model_path = "./中文问答模型"

# 如果还没训练，使用预训练模型
if not os.path.exists(model_path):
    print("⚠️  未找到训练好的模型，使用预训练模型")
    model_path = "bert-base-chinese"

qa_pipeline = pipeline(
    "question-answering",
    model=model_path
)

print(f"✅ 模型加载成功: {model_path}")

# 测试中文问答
print("\n" + "=" * 70)
print("🧪 测试中文问答")
print("=" * 70)

# 示例 1: 地理知识
context1 = """
北京是中华人民共和国的首都，是全国的政治中心、文化中心。北京位于华北平原北部，
背靠燕山，毗邻天津市和河北省。北京有着3000余年的建城史和850余年的建都史，
是世界上拥有世界文化遗产数最多的城市。北京对外开放的旅游景点达200多处，
有世界上最大的皇宫紫禁城、祭天神庙天坛、皇家园林北海公园、颐和园和圆明园，
还有八达岭长城、慕田峪长城以及世界上最大的四合院恭王府等名胜古迹。
"""

questions1 = [
    "北京是什么？",
    "北京位于哪里？",
    "北京有多少年的建都史？",
    "北京有哪些著名景点？"
]

print(f"\n📄 上下文 1: 北京介绍")
print(f"{context1.strip()[:100]}...\n")

for i, question in enumerate(questions1, 1):
    result = qa_pipeline(question=question, context=context1)
    
    print(f"{i}. 问题: {question}")
    print(f"   答案: {result['answer']}")
    print(f"   置信度: {result['score']:.2%}\n")

# 示例 2: 科技知识
context2 = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。可以设想，
未来人工智能带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、
思维的信息过程进行模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
"""

questions2 = [
    "什么是人工智能？",
    "人工智能的研究包括哪些领域？",
    "人工智能能超过人的智能吗？"
]

print("\n" + "=" * 70)
print(f"📄 上下文 2: 人工智能介绍")
print(f"{context2.strip()[:100]}...\n")

for i, question in enumerate(questions2, 1):
    result = qa_pipeline(question=question, context=context2)
    
    print(f"{i}. 问题: {question}")
    print(f"   答案: {result['answer']}")
    print(f"   置信度: {result['score']:.2%}\n")

# 示例 3: 历史知识
context3 = """
长城是中国古代的军事防御工程，是一道高大、坚固而连绵不断的长垣，
用以限隔敌骑的行动。长城不是一道单纯孤立的城墙，而是以城墙为主体，
同大量的城、障、亭、标相结合的防御体系。长城修筑的历史可上溯到西周时期，
发生在首都镐京的著名典故"烽火戏诸侯"就源于此。春秋战国时期列国争霸，
互相防守，长城修筑进入第一个高潮，但此时修筑的长度都比较短。
秦灭六国统一天下后，秦始皇连接和修缮战国长城，始有万里长城之称。
"""

questions3 = [
    "长城是什么？",
    "长城的作用是什么？",
    "谁修建了万里长城？"
]

print("\n" + "=" * 70)
print(f"📄 上下文 3: 长城介绍")
print(f"{context3.strip()[:100]}...\n")

for i, question in enumerate(questions3, 1):
    result = qa_pipeline(question=question, context=context3)
    
    print(f"{i}. 问题: {question}")
    print(f"   答案: {result['answer']}")
    print(f"   置信度: {result['score']:.2%}\n")

# 交互式问答
print("\n" + "=" * 70)
print("💬 交互式问答")
print("=" * 70)
print("""
你可以自己输入上下文和问题进行测试！

示例代码：
```python
from transformers import pipeline

qa = pipeline("question-answering", model="./中文问答模型")

context = "你的上下文..."
question = "你的问题？"

result = qa(question=question, context=context)
print(f"答案: {result['answer']}")
print(f"置信度: {result['score']:.2%}")
```
""")

print("\n" + "=" * 70)
print("✨ 测试完成！")
print("=" * 70)
