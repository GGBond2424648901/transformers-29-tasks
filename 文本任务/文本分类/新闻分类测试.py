#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻分类器测试脚本
加载训练好的模型进行测试
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import pipeline
import json

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, 'output', 'news_classifier')

print("=" * 70)
print("🧪 新闻分类器测试")
print("=" * 70)

# 检查模型是否存在
if not os.path.exists(MODEL_DIR):
    print(f"\n❌ 错误: 模型目录不存在: {MODEL_DIR}")
    print("💡 请先运行 新闻分类训练.py 训练模型")
    exit(1)

# 加载标签映射
label_map_path = os.path.join(MODEL_DIR, 'label_map.json')
with open(label_map_path, 'r', encoding='utf-8') as f:
    LABELS = json.load(f)
    # 转换键为整数
    LABELS = {int(k): v for k, v in LABELS.items()}

print(f"\n📂 模型目录: {MODEL_DIR}")
print(f"📋 类别: {', '.join(LABELS.values())}")

# 加载模型
print("\n🤖 加载模型...")
classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR
)
print("✅ 模型加载成功！")

# 测试样本
test_samples = {
    "科技": [
        "华为发布鸿蒙OS 4.0系统，支持多设备协同",
        "OpenAI推出GPT-5，性能大幅提升",
        "特斯拉发布全自动驾驶系统，安全性提高",
        "量子计算机实现新突破，运算能力提升千倍"
    ],
    "体育": [
        "中国男篮战胜韩国队，晋级亚洲杯决赛",
        "梅西打进职业生涯第700球",
        "东京奥运会中国代表团再夺3金",
        "NBA季后赛勇士队淘汰湖人队"
    ],
    "娱乐": [
        "《流浪地球3》定档春节，预售火爆",
        "周杰伦演唱会门票秒光，粉丝热情高涨",
        "迪士尼新片《冰雪奇缘4》全球首映",
        "著名导演新作入围奥斯卡最佳影片"
    ],
    "财经": [
        "A股三大指数集体收涨，创业板涨超2%",
        "美联储宣布维持利率不变",
        "比特币价格突破6万美元大关",
        "国际油价上涨，布伦特原油涨超3%"
    ],
    "社会": [
        "北京今日最高温达38度，发布高温红色预警",
        "台风杜苏芮登陆福建，多地暴雨",
        "四川发生5.5级地震，暂无人员伤亡",
        "南方多省遭遇洪涝，紧急转移群众"
    ],
    "政治": [
        "全国人大通过新修订的《环境保护法》",
        "国务院发布十四五规划纲要",
        "外交部回应中美关系最新进展",
        "最高法发布司法解释，加强民生保障"
    ]
}

print("\n" + "=" * 70)
print("📊 分类测试结果")
print("=" * 70)

# 统计
total_correct = 0
total_samples = 0

for true_label, texts in test_samples.items():
    print(f"\n【{true_label}】类别测试:")
    print("-" * 70)
    
    correct = 0
    for text in texts:
        result = classifier(text)[0]
        label_id = int(result['label'].split('_')[-1])
        pred_label = LABELS[label_id]
        score = result['score']
        
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
            total_correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {text[:30]}...")
        print(f"   预测: {pred_label} (置信度: {score:.4f})")
        
        total_samples += 1
    
    accuracy = correct / len(texts) * 100
    print(f"\n   准确率: {correct}/{len(texts)} ({accuracy:.1f}%)")

# 总体统计
print("\n" + "=" * 70)
print("📈 总体统计")
print("=" * 70)
overall_accuracy = total_correct / total_samples * 100
print(f"总样本数: {total_samples}")
print(f"正确预测: {total_correct}")
print(f"总体准确率: {overall_accuracy:.2f}%")

# 交互式测试
print("\n" + "=" * 70)
print("💬 交互式测试")
print("=" * 70)
print("输入新闻文本进行分类（输入 'q' 退出）\n")

while True:
    text = input("请输入新闻文本: ").strip()
    
    if text.lower() == 'q':
        print("\n👋 再见！")
        break
    
    if not text:
        continue
    
    try:
        result = classifier(text)[0]
        label_id = int(result['label'].split('_')[-1])
        pred_label = LABELS[label_id]
        score = result['score']
        
        print(f"\n分类结果:")
        print(f"   类别: {pred_label}")
        print(f"   置信度: {score:.4f}")
        
        # 显示所有类别的概率
        results = classifier(text, top_k=len(LABELS))
        print(f"\n   所有类别概率:")
        for r in results:
            label_id = int(r['label'].split('_')[-1])
            label_name = LABELS[label_id]
            print(f"   {label_name}: {r['score']:.4f}")
        print()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")

print("\n✨ 测试完成！")
