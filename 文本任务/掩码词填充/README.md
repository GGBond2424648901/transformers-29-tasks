# 🎭 掩码词填充（Fill-Mask / Masked Language Modeling）

## 📖 任务简介

掩码词填充是一种预训练任务，通过预测被掩码（mask）的词来学习语言表示。BERT 等模型就是通过这种方式进行预训练的。这个任务可以用于词语预测、文本纠错、智能输入法等场景。

## 🎯 应用场景

- **📝 智能输入法**: 词语联想、自动补全
- **🔧 文本纠错**: 拼写检查、语法纠正
- **📚 语言学习**: 填空练习、词汇测试
- **🤖 对话系统**: 句子补全、意图理解
- **📊 数据增强**: 生成相似句子、同义词替换

## 📁 文件说明

- `掩码词填充示例.py` - Pipeline 推理示例
- `README.md` - 本文件

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install transformers torch
```

### 2. 运行示例

```bash
python 掩码词填充示例.py
```

## 💡 使用示例

### 基础填充

```python
from transformers import pipeline

# 创建填充器（中文）
unmasker = pipeline(
    "fill-mask",
    model="bert-base-chinese"
)

# 填充掩码词
text = "今天天气真[MASK]！"
results = unmasker(text)

# 查看结果
for result in results:
    print(f"{result['token_str']}: {result['score']:.2%}")
    print(f"完整句子: {result['sequence']}")
```

### 英文填充

```python
# 英文模型
unmasker_en = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

text = "The weather is [MASK] today."
results = unmasker_en(text)

for result in results:
    print(f"{result['token_str']}: {result['score']:.2%}")
```

### 控制输出数量

```python
# 返回前 10 个预测
results = unmasker(text, top_k=10)

# 只返回最佳预测
best = unmasker(text, top_k=1)[0]
print(f"最佳预测: {best['token_str']}")
```

## 🎨 推荐模型

### 中文模型

| 模型 | 大小 | 特点 |
|------|------|------|
| bert-base-chinese | 中 | 通用中文 BERT |
| hfl/chinese-roberta-wwm-ext | 中 | 全词掩码，效果更好 |
| hfl/chinese-bert-wwm-ext | 中 | 全词掩码 BERT |
| uer/chinese_roberta_L-12_H-768 | 中 | 中文 RoBERTa |

### 英文模型

| 模型 | 大小 | 特点 |
|------|------|------|
| bert-base-uncased | 中 | 通用英文 BERT |
| roberta-base | 中 | 英文 RoBERTa，效果更好 |
| albert-base-v2 | 小 | 轻量级模型 |
| distilbert-base-uncased | 小 | 蒸馏版 BERT，速度快 |

## 💡 使用技巧

### 1. 不同模型的掩码标记

```python
# BERT: [MASK]
text_bert = "今天天气真[MASK]！"

# RoBERTa: <mask>
text_roberta = "今天天气真<mask>！"

# 使用对应的模型
unmasker_bert = pipeline("fill-mask", model="bert-base-chinese")
unmasker_roberta = pipeline("fill-mask", model="hfl/chinese-roberta-wwm-ext")
```

### 2. 多个掩码词

```python
# 注意：一次只能填充一个掩码
# 如果有多个掩码，需要分别处理

text1 = "我喜欢[MASK]。"
text2 = "他是一位[MASK]的科学家。"

result1 = unmasker(text1, top_k=1)
result2 = unmasker(text2, top_k=1)
```

### 3. 句子补全

```python
def complete_sentence(incomplete_text, unmasker):
    """
    自动补全句子
    """
    # 在句子末尾添加掩码
    text_with_mask = incomplete_text + "[MASK]"
    
    # 预测
    results = unmasker(text_with_mask, top_k=5)
    
    # 返回补全的句子
    completions = []
    for result in results:
        completions.append({
            'text': result['sequence'],
            'word': result['token_str'],
            'score': result['score']
        })
    
    return completions

# 使用
incomplete = "人工智能是"
completions = complete_sentence(incomplete, unmasker)

for comp in completions:
    print(f"{comp['text']} ({comp['score']:.2%})")
```

### 4. 文本纠错

```python
def correct_text(text_with_error, correct_word, unmasker):
    """
    将可能错误的词替换为 [MASK]，让模型预测正确的词
    """
    # 替换错误词为掩码
    text_masked = text_with_error.replace(correct_word, "[MASK]")
    
    # 预测
    results = unmasker(text_masked, top_k=5)
    
    # 检查是否包含正确的词
    for result in results:
        if result['token_str'] == correct_word:
            return {
                'is_correct': True,
                'confidence': result['score'],
                'alternatives': results
            }
    
    return {
        'is_correct': False,
        'suggestions': results
    }
```

## 🎯 应用示例

### 1. 智能输入法

```python
class SmartInput:
    def __init__(self, model_name="bert-base-chinese"):
        self.unmasker = pipeline("fill-mask", model=model_name)
    
    def suggest_next_word(self, text, top_k=5):
        """
        预测下一个词
        """
        text_with_mask = text + "[MASK]"
        results = self.unmasker(text_with_mask, top_k=top_k)
        
        suggestions = []
        for result in results:
            word = result['token_str']
            score = result['score']
            suggestions.append((word, score))
        
        return suggestions
    
    def auto_complete(self, partial_word, context=""):
        """
        自动补全
        """
        text = context + partial_word + "[MASK]"
        results = self.unmasker(text, top_k=10)
        
        # 过滤出以 partial_word 开头的词
        completions = []
        for result in results:
            word = result['token_str']
            if word.startswith(partial_word):
                completions.append(word)
        
        return completions

# 使用
smart_input = SmartInput()

# 预测下一个词
suggestions = smart_input.suggest_next_word("今天天气")
print("建议词:", suggestions)

# 自动补全
completions = smart_input.auto_complete("天", "今天")
print("补全:", completions)
```

### 2. 拼写检查

```python
class SpellChecker:
    def __init__(self, model_name="bert-base-chinese"):
        self.unmasker = pipeline("fill-mask", model=model_name)
    
    def check_word(self, sentence, word_position):
        """
        检查指定位置的词是否正确
        """
        words = list(sentence)
        original_word = words[word_position]
        
        # 替换为掩码
        words[word_position] = "[MASK]"
        masked_sentence = "".join(words)
        
        # 预测
        results = self.unmasker(masked_sentence, top_k=10)
        
        # 检查原词是否在预测中
        for i, result in enumerate(results):
            if result['token_str'] == original_word:
                return {
                    'is_correct': True,
                    'confidence': result['score'],
                    'rank': i + 1
                }
        
        # 如果不在预测中，返回建议
        return {
            'is_correct': False,
            'suggestions': [r['token_str'] for r in results[:5]]
        }

# 使用
checker = SpellChecker()
result = checker.check_word("今天天汽真好", 3)  # 检查"汽"字
print(result)
```

### 3. 填空练习生成

```python
class ExerciseGenerator:
    def __init__(self, model_name="bert-base-chinese"):
        self.unmasker = pipeline("fill-mask", model=model_name)
    
    def generate_exercise(self, sentence, num_blanks=1):
        """
        生成填空练习
        """
        import random
        
        words = list(sentence)
        
        # 随机选择要掩码的位置
        positions = random.sample(range(len(words)), num_blanks)
        
        # 保存答案
        answers = []
        for pos in positions:
            answers.append(words[pos])
            words[pos] = "___"
        
        exercise = "".join(words)
        
        return {
            'exercise': exercise,
            'answers': answers,
            'positions': positions
        }
    
    def check_answer(self, exercise, user_answer, position):
        """
        检查答案
        """
        # 将空格替换为掩码
        text = exercise.replace("___", "[MASK]", 1)
        
        # 预测
        results = self.unmasker(text, top_k=10)
        
        # 检查用户答案
        for result in results:
            if result['token_str'] == user_answer:
                return {
                    'correct': True,
                    'confidence': result['score']
                }
        
        return {
            'correct': False,
            'suggestions': [r['token_str'] for r in results[:3]]
        }

# 使用
generator = ExerciseGenerator()

# 生成练习
exercise = generator.generate_exercise("今天天气真好", num_blanks=1)
print(f"练习: {exercise['exercise']}")
print(f"答案: {exercise['answers']}")

# 检查答案
result = generator.check_answer(exercise['exercise'], "好", 0)
print(result)
```

## 📈 性能对比

不同模型在相同任务上的表现：

```python
models = [
    "bert-base-chinese",
    "hfl/chinese-roberta-wwm-ext",
    "hfl/chinese-bert-wwm-ext"
]

test_sentence = "我喜欢[MASK]编程。"

for model_name in models:
    print(f"\n模型: {model_name}")
    unmasker = pipeline("fill-mask", model=model_name)
    results = unmasker(test_sentence, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['token_str']} ({result['score']:.2%})")
```

## ⚠️ 注意事项

1. **掩码标记**: 不同模型使用不同的掩码标记（[MASK] 或 <mask>）
2. **单个掩码**: 一次只能填充一个掩码词
3. **上下文**: 提供足够的上下文信息可以提高预测准确性
4. **模型选择**: 针对特定领域可能需要微调模型
5. **计算资源**: 大模型需要更多内存和计算时间

## 🔗 相关资源

- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [RoBERTa 论文](https://arxiv.org/abs/1907.11692)
- [Hugging Face Fill-Mask 文档](https://huggingface.co/tasks/fill-mask)
- [中文预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)
