#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频分类示例
使用 VideoMAE 进行视频内容分类
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline

print("=" * 70)
print("🎬 视频分类示例")
print("=" * 70)

# 创建视频分类 pipeline
classifier = pipeline(
    "video-classification",
    model="MCG-NJU/videomae-base-finetuned-kinetics"
)

print("✅ 模型加载成功！")
print("""
应用场景：
- 📹 视频内容审核
- 🎯 行为识别
- 🏃 动作分类
- 📊 视频标注

使用方法：
```python
# 分类视频
result = classifier("video.mp4")
print(result)
# [{'label': 'playing basketball', 'score': 0.95}]

# 批量分类
videos = ["video1.mp4", "video2.mp4"]
results = classifier(videos)
```

注意事项：
1. 视频文件需要是常见格式（mp4, avi等）
2. 模型会自动采样视频帧
3. 较长视频会被截断或采样
""")

print("\n✨ 示例完成！")
