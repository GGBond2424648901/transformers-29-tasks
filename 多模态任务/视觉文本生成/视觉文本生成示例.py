#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉+文本生成示例
基于图像和文本的多模态生成
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

print("=" * 70)
print("👁️✍️ 视觉+文本生成示例")
print("=" * 70)

# 创建图像文本生成 pipeline
generator = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

print("✅ 模型加载成功！")

# 加载图像
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(BytesIO(requests.get(image_url).content))

# 生成描述
prompt = "Describe this image in detail."
result = generator(image, prompt=prompt)

print(f"\n📸 图像已加载")
print(f"💬 提示: {prompt}")
print(f"📝 生成: {result[0]['generated_text']}")

print("""
\n应用场景：
- 🤖 AI 助手 - 图像理解与对话
- 📝 内容创作 - 图像配文
- 🎨 图像编辑 - 指令式编辑
- 📚 教育 - 图像讲解

功能示例：

1. 图像描述
```python
prompt = "What do you see in this image?"
result = generator(image, prompt=prompt)
```

2. 图像问答
```python
prompt = "How many people are in the image?"
result = generator(image, prompt=prompt)
```

3. 图像编辑指令
```python
prompt = "How would you edit this image to make it brighter?"
result = generator(image, prompt=prompt)
```

4. 创意写作
```python
prompt = "Write a short story based on this image."
result = generator(image, prompt=prompt)
```

推荐模型：
- llava-hf/llava-1.5-7b-hf: LLaVA 模型
- Salesforce/blip2-opt-2.7b: BLIP-2
- Qwen/Qwen-VL-Chat: Qwen-VL
- liuhaotian/llava-v1.6-vicuna-7b: LLaVA 1.6

使用技巧：
1. 提示词要清晰具体
2. 可以进行多轮对话
3. 支持复杂的推理任务
""")

print("\n✨ 示例完成！")
