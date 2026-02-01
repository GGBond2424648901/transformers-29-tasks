#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 RoBERTa 模型是否可以加载
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

print("=" * 70)
print("🔍 检查 RoBERTa 模型")
print("=" * 70)

# 检查模型文件
model_path = r"D:\transformers训练\transformers-main\预训练模型下载处\hub\models--hfl--chinese-roberta-wwm-ext"
print(f"\n📁 模型路径: {model_path}")

import os
if os.path.exists(model_path):
    print("✅ 模型文件夹存在")
    
    # 列出文件
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(('.bin', '.safetensors', '.json', '.txt')):
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)
                print(f"   {file}: {size / 1024 / 1024:.1f} MB")
else:
    print("❌ 模型文件夹不存在")

# 尝试加载
print("\n" + "=" * 70)
print("🔄 尝试加载模型")
print("=" * 70)

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

print(f"\n📊 PyTorch 版本: {torch.__version__}")
print(f"📊 CUDA 可用: {torch.cuda.is_available()}")

model_name = "hfl/chinese-roberta-wwm-ext"

# 方法 1: 默认加载
print(f"\n方法 1: 默认加载")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("✅ 成功！可以使用 RoBERTa")
    print(f"   模型类型: {type(model).__name__}")
except Exception as e:
    print(f"❌ 失败: {str(e)[:200]}")

# 方法 2: 使用 use_safetensors=False
print(f"\n方法 2: 强制使用 pytorch_model.bin")
try:
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        use_safetensors=False
    )
    print("✅ 成功！")
except Exception as e:
    print(f"❌ 失败: {str(e)[:200]}")

# 方法 3: 使用 trust_remote_code
print(f"\n方法 3: 使用 trust_remote_code")
try:
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✅ 成功！")
except Exception as e:
    print(f"❌ 失败: {str(e)[:200]}")

print("\n" + "=" * 70)
print("📝 结论")
print("=" * 70)

print("""
如果所有方法都失败，说明需要升级 PyTorch：

升级命令：
pip install --upgrade torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

升级后重新运行此脚本检查。
""")
