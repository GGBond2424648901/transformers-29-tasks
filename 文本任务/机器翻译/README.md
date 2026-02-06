# 🌍 机器翻译（Machine Translation）

## 📖 任务简介

机器翻译是将文本从一种语言自动翻译成另一种语言的任务。现代机器翻译主要使用神经网络模型（NMT），能够生成流畅、准确的译文。

## 🎯 应用场景

- **🌐 跨语言交流**: 实时翻译聊天、邮件
- **📚 文档翻译**: 技术文档、合同、论文翻译
- **🎬 字幕翻译**: 视频、电影字幕
- **🛍️ 电商国际化**: 商品描述多语言展示
- **📱 应用本地化**: 软件界面多语言支持

## 📁 文件说明

- `run_translation.py` - 使用 Trainer API 训练翻译模型
- `run_translation_no_trainer.py` - 不使用 Trainer 训练
- `requirements.txt` - 依赖包列表
- `README.md` - 本文件

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练

#### 使用 WMT 数据集

```bash
python run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-zh \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en \
    --target_lang zh \
    --output_dir ./output \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3 \
    --predict_with_generate
```

#### 使用本地数据

```bash
python run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-zh \
    --train_file train.json \
    --validation_file val.json \
    --source_lang en \
    --target_lang zh \
    --output_dir ./output \
    --do_train \
    --do_eval
```

## 💡 使用示例

### Pipeline 推理

```python
from transformers import pipeline

# 英译中
translator_en_zh = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-zh"
)

result = translator_en_zh("Hello, how are you?")
print(result[0]['translation_text'])
# 输出：你好，你好吗？

# 中译英
translator_zh_en = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en"
)

result = translator_zh_en("今天天气真好！")
print(result[0]['translation_text'])
# 输出：The weather is really nice today!
```

### 批量翻译

```python
texts = [
    "Good morning!",
    "How are you?",
    "Nice to meet you."
]

translations = translator_en_zh(texts)

for text, trans in zip(texts, translations):
    print(f"{text} -> {trans['translation_text']}")
```

### 训练自定义翻译模型

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# 1. 加载数据集
dataset = load_dataset("wmt16", "ro-en")

# 2. 加载模型和分词器
model_name = "Helsinki-NLP/opus-mt-en-ro"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. 数据预处理
source_lang = "en"
target_lang = "ro"

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True
    )
    
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)

# 4. 训练配置
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    predict_with_generate=True,
)

# 5. 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# 6. 创建 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# 7. 开始训练
trainer.train()
```

## 🎨 推荐模型

### Helsinki-NLP OPUS-MT 系列

| 模型 | 语言对 | 特点 |
|------|--------|------|
| Helsinki-NLP/opus-mt-en-zh | 英→中 | 轻量级，速度快 |
| Helsinki-NLP/opus-mt-zh-en | 中→英 | 轻量级，速度快 |
| Helsinki-NLP/opus-mt-en-de | 英→德 | 欧洲语言效果好 |
| Helsinki-NLP/opus-mt-en-fr | 英→法 | 欧洲语言效果好 |

### 其他模型

| 模型 | 特点 |
|------|------|
| facebook/mbart-large-50-many-to-many-mmt | 多语言互译，支持 50 种语言 |
| google/mt5-base | 多语言 T5，支持翻译任务 |
| facebook/nllb-200-distilled-600M | 支持 200 种语言 |

## 📊 数据格式

### JSON 格式

```json
[
    {
        "en": "Hello, world!",
        "zh": "你好，世界！"
    },
    {
        "en": "Good morning.",
        "zh": "早上好。"
    }
]
```

### CSV 格式

```csv
source,target
"Hello, world!","你好，世界！"
"Good morning.","早上好。"
```

## ⚙️ 重要参数

### 生成参数

```python
translator(
    text,
    max_length=128,        # 最大翻译长度
    num_beams=5,           # Beam search 数量
    early_stopping=True,   # 早停
    length_penalty=1.0     # 长度惩罚
)
```

### 训练参数

```bash
--source_lang en               # 源语言
--target_lang zh               # 目标语言
--max_source_length 128        # 源文本最大长度
--max_target_length 128        # 译文最大长度
--num_beams 5                  # Beam search
--predict_with_generate        # 使用生成模式评估
```

## 💡 使用技巧

### 1. 多语言翻译

```python
# 使用 mBART 进行多语言翻译
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)

# 中文翻译成英文
tokenizer.src_lang = "zh_CN"
encoded = tokenizer("今天天气真好！", return_tensors="pt")
generated_tokens = model.generate(
    **encoded,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(translation[0])
```

### 2. 处理长文本

```python
def translate_long_text(text, max_length=500):
    # 按句子分割
    sentences = text.split('。')
    
    # 分批翻译
    translations = []
    for sentence in sentences:
        if sentence.strip():
            result = translator(sentence + '。')
            translations.append(result[0]['translation_text'])
    
    return ' '.join(translations)
```

### 3. 提高翻译质量

```python
# 使用更多的 beam
result = translator(
    text,
    num_beams=10,          # 增加 beam 数量
    length_penalty=1.2,    # 调整长度惩罚
    early_stopping=True
)

# 使用采样
result = translator(
    text,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
```

## 📈 评估指标

### BLEU 分数

```python
from datasets import load_metric

bleu = load_metric("sacrebleu")

predictions = ["The weather is nice today."]
references = [["Today's weather is good."]]

results = bleu.compute(
    predictions=predictions,
    references=references
)

print(f"BLEU: {results['score']:.2f}")
```

### 其他指标

- **BLEU**: 最常用的机器翻译评估指标
- **METEOR**: 考虑同义词和词干
- **TER**: 翻译编辑率
- **chrF**: 字符级 F-score

## 🎯 应用示例

### 1. 实时聊天翻译

```python
def chat_translator(message, source_lang="en", target_lang="zh"):
    # 加载对应的翻译模型
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    translator = pipeline("translation", model=model_name)
    
    result = translator(message)
    return result[0]['translation_text']

# 使用
user_message = "Hello, how can I help you?"
translated = chat_translator(user_message, "en", "zh")
print(translated)
```

### 2. 文档翻译

```python
def translate_document(file_path, source_lang="en", target_lang="zh"):
    # 读取文档
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按段落分割
    paragraphs = content.split('\n\n')
    
    # 翻译每个段落
    translator = pipeline(
        "translation",
        model=f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    )
    
    translated_paragraphs = []
    for para in paragraphs:
        if para.strip():
            result = translator(para)
            translated_paragraphs.append(result[0]['translation_text'])
    
    # 保存翻译结果
    output_path = file_path.replace('.txt', f'_{target_lang}.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(translated_paragraphs))
    
    return output_path
```

### 3. 字幕翻译

```python
def translate_subtitles(srt_file, source_lang="en", target_lang="zh"):
    import pysrt
    
    # 读取字幕
    subs = pysrt.open(srt_file)
    
    # 翻译
    translator = pipeline(
        "translation",
        model=f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    )
    
    for sub in subs:
        result = translator(sub.text)
        sub.text = result[0]['translation_text']
    
    # 保存
    output_file = srt_file.replace('.srt', f'_{target_lang}.srt')
    subs.save(output_file, encoding='utf-8')
    
    return output_file
```

## ⚠️ 注意事项

1. **语言对**: 不同语言对的翻译质量差异很大
2. **专业术语**: 专业领域可能需要微调模型
3. **文化差异**: 翻译需要考虑文化背景
4. **文本长度**: 注意模型的最大长度限制
5. **计算资源**: 大模型需要 GPU 加速

## 🔗 相关资源

- [OPUS-MT 模型集合](https://huggingface.co/Helsinki-NLP)
- [mBART 论文](https://arxiv.org/abs/2001.08210)
- [WMT 翻译竞赛](https://www.statmt.org/wmt21/)
- [BLEU 评估指标](https://en.wikipedia.org/wiki/BLEU)
