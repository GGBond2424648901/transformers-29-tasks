#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本转语音（TTS）实战示例
使用 Bark 模型生成语音
"""

import os

# 设置模型缓存路径
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np

print("=" * 70)
print("🎤 文本转语音（TTS）实战示例")
print("=" * 70)
print(f"📁 模型缓存路径: {os.environ['HF_HOME']}")
print("=" * 70)

# 1. 创建 TTS pipeline
print("\n📦 步骤 1: 加载模型")
print("-" * 70)
print("⚠️  注意：Bark 模型较大，首次下载需要时间")

synthesizer = pipeline(
    "text-to-speech",
    model="suno/bark-small"
)

print("✅ 模型加载成功！")
print(f"   模型: suno/bark-small")
print(f"   任务: 文本转语音")

# 2. 生成语音
print("\n🎵 步骤 2: 生成语音")
print("-" * 70)

text = "你好，欢迎使用 Transformers 文本转语音功能！"
print(f"输入文本: {text}")

speech = synthesizer(text)

print("✅ 语音生成完成！")
print(f"\n📊 音频信息:")
print(f"   采样率: {speech['sampling_rate']} Hz")
print(f"   音频长度: {len(speech['audio'])} 采样点")
print(f"   时长: {len(speech['audio']) / speech['sampling_rate']:.2f} 秒")

# 3. 保存音频文件
print("\n💾 步骤 3: 保存音频文件")
print("-" * 70)

output_file = "生成的语音.wav"
wavfile.write(
    output_file,
    rate=speech['sampling_rate'],
    data=speech['audio']
)

print(f"✅ 音频已保存到: {output_file}")

# 4. 批量生成
print("\n📦 步骤 4: 批量生成语音")
print("-" * 70)

texts = [
    "早上好！",
    "今天天气真不错。",
    "祝你有美好的一天！"
]

print(f"生成 {len(texts)} 段语音...")

for i, text in enumerate(texts, 1):
    print(f"\n   {i}. 生成: {text}")
    speech = synthesizer(text)
    
    output_file = f"语音_{i}.wav"
    wavfile.write(
        output_file,
        rate=speech['sampling_rate'],
        data=speech['audio']
    )
    print(f"      ✅ 已保存到: {output_file}")

# 5. 高级用法
print("\n" + "=" * 70)
print("💡 高级用法")
print("=" * 70)
print("""
Bark 模型支持多种语言和说话风格：

1. 多语言支持：
   - 英语、中文、法语、德语等
   - 自动检测语言

2. 情感和语调：
   - 在文本中添加标点符号影响语调
   - 使用 [laughs]、[sighs] 等标记添加情感

3. 说话人选择：
   - 可以指定不同的说话人声音
   - 支持男声、女声等

示例代码：

# 带情感的语音
text = "哇！[laughs] 这真是太棒了！"
speech = synthesizer(text)

# 英文语音
text = "Hello, how are you today?"
speech = synthesizer(text)
""")

# 6. 应用场景
print("\n" + "=" * 70)
print("🎯 应用场景")
print("=" * 70)
print("""
文本转语音的主要应用：

1. 📚 有声书制作
   - 自动朗读文本
   - 多角色配音
   - 批量生成

2. 🗣️ 语音助手
   - 智能客服
   - 导航语音
   - 提醒通知

3. 🎬 视频配音
   - 解说词生成
   - 多语言配音
   - 快速原型

4. ♿ 无障碍辅助
   - 屏幕阅读
   - 文字转语音
   - 帮助视障人士

5. 🎓 教育培训
   - 课程讲解
   - 语言学习
   - 在线教学
""")

# 7. 性能优化建议
print("\n" + "=" * 70)
print("⚡ 性能优化建议")
print("=" * 70)
print("""
1. 使用更小的模型：
   - bark-small: 快速，质量较好
   - bark: 质量最好，但速度慢

2. GPU 加速：
   - 确保安装了 CUDA 版本的 PyTorch
   - 模型会自动使用 GPU

3. 批量处理：
   - 一次生成多段语音
   - 减少模型加载开销

4. 文本预处理：
   - 分段处理长文本
   - 避免单次生成过长语音
""")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
print("\n💡 提示：生成的音频文件可以用任何音频播放器打开")
