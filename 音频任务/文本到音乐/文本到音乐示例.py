#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本到音乐生成示例
使用 MusicGen 根据文本描述生成音乐
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
import scipy.io.wavfile as wavfile

print("=" * 70)
print("🎵 文本到音乐生成示例")
print("=" * 70)

# 创建音乐生成器
synthesizer = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small"
)

print("✅ 模型加载成功！")

# 生成音乐
text = "upbeat electronic dance music with a catchy melody"
print(f"\n输入描述: {text}")

music = synthesizer(text, forward_params={"max_new_tokens": 256})

# 保存音乐
output_file = "generated_music.wav"
wavfile.write(
    output_file,
    rate=music["sampling_rate"],
    data=music["audio"][0]
)

print(f"✅ 音乐已保存到: {output_file}")

print("""
\n应用场景：
- 🎬 背景音乐生成
- 🎮 游戏音效
- 📹 视频配乐
- 🎨 创意音乐制作

使用技巧：
1. 描述要具体（风格、节奏、情绪）
2. 可以指定乐器类型
3. 支持多种音乐风格

示例描述：
- "calm piano music for meditation"
- "energetic rock music with guitar solo"
- "ambient electronic music for studying"
""")

print("\n✨ 示例完成！")
