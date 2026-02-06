#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试微调后的 ChatGLM 客服模型
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

print("=" * 70)
print("🧪 测试 ChatGLM 客服模型")
print("=" * 70)

# ============================================================================
# 加载模型
# ============================================================================

def load_model(base_model="THUDM/chatglm-6b", lora_path="output/chatglm-customer-lora"):
    """加载基础模型和 LoRA 权重"""
    
    print("\n📦 加载模型...")
    
    try:
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        
        # 加载基础模型
        model = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 加载 LoRA 权重
        if os.path.exists(lora_path):
            print(f"✅ 加载 LoRA 权重: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
        else:
            print(f"⚠️  未找到 LoRA 权重，使用基础模型")
        
        model = model.eval()
        
        print("✅ 模型加载成功")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

# ============================================================================
# 测试对话
# ============================================================================

def test_chat(model, tokenizer):
    """测试客服对话"""
    
    print("\n" + "=" * 70)
    print("💬 客服对话测试")
    print("=" * 70)
    
    # 测试问题列表
    test_questions = [
        "如何退货？",
        "发货需要多久？",
        "支持哪些支付方式？",
        "可以修改收货地址吗？",
        "如何联系客服？",
        "会员有什么权益？",
        "商品可以换货吗？",
        "如何查询物流？",
    ]
    
    history = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"问题 {i}: {question}")
        print("-" * 70)
        
        try:
            response, history = model.chat(
                tokenizer,
                f"用户问：{question}",
                history=[]  # 每次清空历史，独立对话
            )
            
            print(f"回答: {response}")
            
        except Exception as e:
            print(f"❌ 对话失败: {e}")
    
    print("\n" + "=" * 70)

# ============================================================================
# 交互式测试
# ============================================================================

def interactive_test(model, tokenizer):
    """交互式测试"""
    
    print("\n" + "=" * 70)
    print("🎮 交互式测试模式")
    print("=" * 70)
    print("输入问题进行测试，输入 'quit' 退出")
    print("-" * 70)
    
    history = []
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            # 添加"用户问："前缀
            question = f"用户问：{user_input}"
            
            response, history = model.chat(
                tokenizer,
                question,
                history=[]  # 单轮对话
            )
            
            print(f"\n客服: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")

# ============================================================================
# 对比测试
# ============================================================================

def compare_models(base_model_name="THUDM/chatglm-6b", lora_path="output/chatglm-customer-lora"):
    """对比原始模型和微调模型"""
    
    print("\n" + "=" * 70)
    print("📊 模型对比测试")
    print("=" * 70)
    
    # 加载原始模型
    print("\n1️⃣ 加载原始 ChatGLM-6B...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        ).eval()
        print("✅ 原始模型加载成功")
    except Exception as e:
        print(f"❌ 原始模型加载失败: {e}")
        return
    
    # 加载微调模型
    print("\n2️⃣ 加载微调模型...")
    try:
        finetuned_model = PeftModel.from_pretrained(base_model, lora_path).eval()
        print("✅ 微调模型加载成功")
    except Exception as e:
        print(f"❌ 微调模型加载失败: {e}")
        return
    
    # 测试问题
    test_question = "用户问：如何退货？"
    
    print("\n" + "=" * 70)
    print(f"测试问题: {test_question}")
    print("=" * 70)
    
    # 原始模型回答
    print("\n【原始模型回答】")
    print("-" * 70)
    try:
        response1, _ = base_model.chat(tokenizer, test_question, history=[])
        print(response1)
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 微调模型回答
    print("\n【微调模型回答】")
    print("-" * 70)
    try:
        response2, _ = finetuned_model.chat(tokenizer, test_question, history=[])
        print(response2)
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    print("\n" + "=" * 70)
    print("💡 观察两个模型的回答差异")
    print("   微调模型应该更符合客服风格，回答更规范")
    print("=" * 70)

# ============================================================================
# 主函数
# ============================================================================

def main():
    import sys
    
    # 检查是否有 LoRA 权重
    lora_path = "output/chatglm-customer-lora"
    
    if not os.path.exists(lora_path):
        print("\n⚠️  未找到 LoRA 权重")
        print(f"   路径: {lora_path}")
        print("\n请先运行训练脚本：")
        print("   python chatglm_lora_finetune.py")
        print("   或双击 开始训练.bat")
        return
    
    # 加载模型
    model, tokenizer = load_model(lora_path=lora_path)
    
    if model is None:
        return
    
    # 选择测试模式
    print("\n" + "=" * 70)
    print("选择测试模式：")
    print("=" * 70)
    print("1. 自动测试（预设问题）")
    print("2. 交互式测试（手动输入）")
    print("3. 对比测试（原始 vs 微调）")
    print("4. 全部测试")
    print("=" * 70)
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        test_chat(model, tokenizer)
    elif choice == "2":
        interactive_test(model, tokenizer)
    elif choice == "3":
        compare_models()
    elif choice == "4":
        test_chat(model, tokenizer)
        print("\n" + "=" * 70)
        input("按 Enter 继续交互式测试...")
        interactive_test(model, tokenizer)
    else:
        print("❌ 无效选择，默认运行自动测试")
        test_chat(model, tokenizer)
    
    print("\n" + "=" * 70)
    print("✨ 测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
