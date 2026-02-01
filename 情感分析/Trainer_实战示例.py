"""
Trainer API 实战示例
演示如何使用 Trainer 训练一个简单的文本分类模型
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np

print("=" * 70)
print("🚀 Trainer API 实战示例：情感分类")
print("=" * 70)

# ============================================================================
# 步骤 1: 准备数据
# ============================================================================
print("\n📊 步骤 1: 准备训练数据")
print("-" * 70)

# 创建一个简单的情感分类数据集
train_texts = [
    "这个产品太棒了，我非常喜欢！",
    "质量很差，完全不值这个价格。",
    "还可以，符合预期。",
    "非常满意，会推荐给朋友。",
    "太失望了，浪费钱。",
    "性价比很高，值得购买。",
    "不推荐，有很多问题。",
    "超出预期，非常好用！",
    "一般般，没什么特别的。",
    "完美！正是我想要的。",
] * 10  # 重复 10 次以增加数据量

train_labels = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1] * 10  # 1=正面, 0=负面

eval_texts = [
    "很好用，推荐购买。",
    "不太满意，有待改进。",
    "物超所值！",
    "质量一般。",
]
eval_labels = [1, 0, 1, 1]

print(f"✅ 训练样本数: {len(train_texts)}")
print(f"✅ 评估样本数: {len(eval_texts)}")
print(f"✅ 示例文本: {train_texts[0]}")
print(f"✅ 示例标签: {train_labels[0]} (1=正面, 0=负面)")

# ============================================================================
# 步骤 2: 加载分词器和模型
# ============================================================================
print("\n🤖 步骤 2: 加载模型和分词器")
print("-" * 70)

# 使用中文 BERT 模型（如果网络不好，可以换成 bert-base-uncased）
model_name = "bert-base-chinese"  # 或 "bert-base-uncased"

print(f"正在加载模型: {model_name}")
print("⏳ 首次运行会下载模型，请稍候...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 二分类：正面/负面
    )
    print(f"✅ 模型加载成功！")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("💡 提示：如果下载失败，可以：")
    print("   1. 使用镜像站：export HF_ENDPOINT=https://hf-mirror.com")
    print("   2. 或使用更小的模型：distilbert-base-uncased")
    exit(1)

# ============================================================================
# 步骤 3: 数据预处理
# ============================================================================
print("\n🔧 步骤 3: 数据预处理（分词）")
print("-" * 70)

def tokenize_function(examples):
    """将文本转换为模型输入格式"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# 创建 Dataset 对象
train_dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels
})

eval_dataset = Dataset.from_dict({
    "text": eval_texts,
    "label": eval_labels
})

# 应用分词
print("正在对数据进行分词...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

print(f"✅ 分词完成！")
print(f"   训练集特征: {train_dataset.column_names}")
print(f"   示例输入: {train_dataset[0]['input_ids'][:10]}...")

# ============================================================================
# 步骤 4: 定义评估指标
# ============================================================================
print("\n📈 步骤 4: 定义评估指标")
print("-" * 70)

def compute_metrics(eval_pred):
    """计算准确率"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

print("✅ 评估指标: 准确率（Accuracy）")

# ============================================================================
# 步骤 5: 配置训练参数
# ============================================================================
print("\n⚙️  步骤 5: 配置训练参数")
print("-" * 70)

# 检查 GPU 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = torch.cuda.is_available()  # 只在 GPU 上使用 FP16

# 获取脚本所在目录
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "my_sentiment_model")
logging_dir = os.path.join(script_dir, "logs")

training_args = TrainingArguments(
    # 基础设置
    output_dir=output_dir,                    # 输出目录（绝对路径）
    
    # 训练设置
    num_train_epochs=3,                       # 训练 3 轮
    per_device_train_batch_size=8,            # 训练批次大小
    per_device_eval_batch_size=16,            # 评估批次大小
    learning_rate=2e-5,                       # 学习率
    weight_decay=0.01,                        # 权重衰减
    
    # 性能优化
    fp16=use_fp16,                            # 混合精度（GPU）
    
    # 日志和保存
    logging_dir=logging_dir,                  # 日志目录（绝对路径）
    logging_steps=10,                         # 每 10 步记录一次
    save_strategy="epoch",                    # 每个 epoch 保存一次
    save_total_limit=2,                       # 最多保存 2 个检查点
    
    # 评估设置
    eval_strategy="epoch",                    # 每个 epoch 评估一次
    load_best_model_at_end=True,              # 训练结束加载最佳模型
    metric_for_best_model="accuracy",         # 使用准确率选择最佳模型
    
    # 其他
    seed=42,                                  # 随机种子
    report_to="none",                         # 不上报到外部服务
)

print(f"✅ 训练配置:")
print(f"   设备: {device}")
print(f"   训练轮数: {training_args.num_train_epochs}")
print(f"   批次大小: {training_args.per_device_train_batch_size}")
print(f"   学习率: {training_args.learning_rate}")
print(f"   混合精度: {training_args.fp16}")

# ============================================================================
# 步骤 6: 创建 Trainer
# ============================================================================
print("\n🎯 步骤 6: 创建 Trainer")
print("-" * 70)

trainer = Trainer(
    model=model,                              # 模型
    args=training_args,                       # 训练参数
    train_dataset=train_dataset,              # 训练数据
    eval_dataset=eval_dataset,                # 评估数据
    compute_metrics=compute_metrics,          # 评估指标
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # 数据整理器
    processing_class=tokenizer,               # 处理类（新版本用这个代替 tokenizer）
)

print("✅ Trainer 创建成功！")

# ============================================================================
# 步骤 7: 开始训练
# ============================================================================
print("\n🚀 步骤 7: 开始训练")
print("=" * 70)
print("⏳ 训练中，请稍候...")
print()

try:
    # 训练模型（就这一行！）
    train_result = trainer.train()
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"训练时间: {train_result.metrics.get('train_runtime', 0):.2f} 秒")
    if 'train_samples' in train_result.metrics:
        print(f"训练样本数: {train_result.metrics['train_samples']}")
    print(f"训练步数: {train_result.metrics.get('train_steps', 0)}")
    print(f"最终损失: {train_result.metrics.get('train_loss', 0):.4f}")
    
except Exception as e:
    print(f"\n❌ 训练失败: {e}")
    exit(1)

# ============================================================================
# 步骤 8: 评估模型
# ============================================================================
print("\n📊 步骤 8: 评估模型")
print("=" * 70)

eval_result = trainer.evaluate()

print("✅ 评估完成！")
print(f"   准确率: {eval_result['eval_accuracy']:.2%}")
print(f"   损失: {eval_result['eval_loss']:.4f}")

# ============================================================================
# 步骤 9: 保存模型
# ============================================================================
print("\n💾 步骤 9: 保存模型")
print("=" * 70)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ 模型已保存到: {output_dir}")

# ============================================================================
# 步骤 10: 测试模型
# ============================================================================
print("\n🧪 步骤 10: 测试模型")
print("=" * 70)

# 使用 pipeline 进行推理
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=output_dir,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

test_texts = [
    "这个产品真的很棒！",
    "太糟糕了，不推荐。",
    "还行吧，一般般。",
]

print("测试样本预测结果：")
for text in test_texts:
    result = classifier(text)[0]
    label = "正面 😊" if result['label'] == 'LABEL_1' else "负面 😞"
    print(f"   文本: {text}")
    print(f"   预测: {label} (置信度: {result['score']:.2%})")
    print()

# ============================================================================
# 总结
# ============================================================================
print("=" * 70)
print("✨ 总结")
print("=" * 70)
print("""
恭喜！你已经成功使用 Trainer API 训练了一个情感分类模型！

🎯 你学到了什么：
1. ✅ 如何准备训练数据
2. ✅ 如何加载预训练模型
3. ✅ 如何配置训练参数
4. ✅ 如何使用 Trainer 训练模型
5. ✅ 如何评估和保存模型
6. ✅ 如何使用训练好的模型

💡 关键优势：
- 只需 ~100 行代码（包含注释）
- 自动处理 GPU、混合精度、日志等
- 代码清晰易懂，易于维护

🚀 下一步：
- 尝试更大的数据集
- 调整超参数（学习率、批次大小等）
- 尝试不同的模型（RoBERTa、ALBERT 等）
- 添加更多评估指标（F1、Precision、Recall）
- 使用真实数据集（如 IMDB、SST-2）

📚 更多资源：
- 官方文档: https://huggingface.co/docs/transformers/training
- 示例代码: examples/pytorch/text-classification/
- 中文文档: docs/source/zh/training.md
""")
print("=" * 70)
