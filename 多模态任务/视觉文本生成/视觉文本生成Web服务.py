#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉文本生成 Web 服务（两步法）
步骤1: BLIP 生成图像描述
步骤2: Qwen 根据描述生成故事
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

# 尝试导入翻译库
try:
    from googletrans import Translator as GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

app = Flask(__name__)

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🚀 正在启动视觉文本生成 Web 服务（两步法）...")
print("=" * 70)

# 加载 BLIP 模型
print("📦 步骤 1/2: 加载 BLIP 图像描述模型...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# 检测是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model = blip_model.to(device)
print(f"✅ BLIP 模型加载成功！(设备: {device})")

# 加载 Qwen 模型
print("📦 步骤 2/2: 加载 Qwen2.5 故事生成模型...")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
qwen_model = qwen_model.to(device)
print(f"✅ Qwen 模型加载成功！(设备: {device})")

# 初始化翻译器
translator = None
if TRANSLATOR_AVAILABLE:
    print("📦 初始化 Google 翻译...")
    try:
        translator = GoogleTranslator()
        print("✅ Google 翻译初始化成功！")
    except Exception as e:
        print(f"⚠️  Google 翻译初始化失败: {e}")
        translator = None

def generate_story_with_qwen(image_description):
    """使用 Qwen 根据图像描述生成中文故事"""
    prompt = f"""你是一位富有创意的作家。请根据以下图像描述，创作一个生动有趣的短篇故事（300-500字）。

图像描述：{image_description}

要求：
1. 故事要有完整的情节（开头、发展、高潮、结尾）
2. 包含生动的人物描写和场景描绘
3. 富有想象力和情感
4. 语言优美流畅，有文学性
5. 可以适当发挥想象，但要基于图像描述

请直接开始讲故事，不要有任何前缀说明："""
    
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=600,
        temperature=0.85,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    story = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return story.strip()

# HTML 模板（简化版，保留核心功能）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎨 AI 看图讲故事</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background-image: url('/static/background');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
            padding: 20px;
            overflow-y: auto;
            overflow-x: hidden;
        }
        
        .falling-item {
            position: fixed;
            font-size: 24px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.7;
        }
        
        @keyframes fall {
            0% { transform: translateY(-10px) rotate(0deg); opacity: 0.7; }
            100% { transform: translateY(100vh) rotate(360deg); opacity: 0.2; }
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 1000px;
            margin: 20px auto;
            position: relative;
            z-index: 10;
        }
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .upload-area {
            border: 3px dashed #3498db;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: rgba(52, 152, 219, 0.05);
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: rgba(52, 152, 219, 0.1); }
        .upload-icon { font-size: 4em; margin-bottom: 15px; }
        .upload-text { font-size: 1.2em; color: #34495e; margin-bottom: 10px; }
        #fileInput { display: none; }
        .preview-container { display: none; margin-bottom: 20px; }
        .preview-image {
            max-width: 100%;
            max-height: 350px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            display: block;
            margin: 0 auto 20px;
        }
        .button-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 20px 0;
        }
        button {
            padding: 12px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-2px); }
        .btn-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .result-container {
            display: none;
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
        }
        .result-title {
            font-size: 1.3em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .result-content {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            line-height: 1.8;
        }
        .result-chinese {
            color: #2c3e50;
            font-size: 1.1em;
            margin-bottom: 15px;
            white-space: pre-wrap;
        }
        .result-english {
            color: #7f8c8d;
            font-size: 0.95em;
            font-style: italic;
            padding-top: 15px;
            border-top: 2px dashed #ddd;
            white-space: pre-wrap;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid rgba(102, 126, 234, 0.1);
            border-left-color: #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text {
            color: #667eea;
            font-size: 1.1em;
            font-weight: 600;
        }
        .info-box {
            background: linear-gradient(135deg, #4facfe15 0%, #00f2fe15 100%);
            border-left: 4px solid #4facfe;
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .info-box p {
            color: #34495e;
            line-height: 1.8;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 AI 视觉文本生成</h1>
        <p class="subtitle">上传图片，选择模式：简单描述 或 创作故事</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">🖼️</div>
            <div class="upload-text">点击或拖拽图片到这里</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="preview-container" id="previewContainer">
            <img id="previewImage" class="preview-image" alt="预览图片">
            
            <div style="background: rgba(52, 152, 219, 0.05); border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                <div style="font-size: 1.1em; color: #2c3e50; margin-bottom: 15px; font-weight: 600;">
                    � 选择生成模式：
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <button class="mode-btn active" data-mode="describe" style="padding: 15px; background: white; border: 2px solid #3498db; border-radius: 8px; color: #3498db; cursor: pointer; font-size: 1em; font-weight: 600; transition: all 0.3s;">
                        📝 简单描述
                    </button>
                    <button class="mode-btn" data-mode="story" style="padding: 15px; background: white; border: 2px solid #3498db; border-radius: 8px; color: #3498db; cursor: pointer; font-size: 1em; font-weight: 600; transition: all 0.3s;">
                        📖 创作故事
                    </button>
                </div>
                <div style="margin-top: 10px; color: #7f8c8d; font-size: 0.9em; text-align: center;" id="modeHint">
                    快速生成图片的基本描述
                </div>
            </div>
            
            <div class="button-group">
                <button class="btn-primary" id="generateBtn">✨ 开始生成</button>
                <button class="btn-success" id="changeImageBtn">🔄 更换图片</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">AI 正在创作故事中...</div>
        </div>
        
        <div class="result-container" id="resultContainer">
            <div class="result-title">✨ AI 创作的故事：</div>
            <div class="result-content">
                <div class="result-chinese" id="resultChinese"></div>
                <div class="result-english" id="resultEnglish"></div>
            </div>
        </div>
        
        <div class="info-box">
            <p><strong>🤖 技术：</strong>BLIP 图像理解 + Qwen2.5 故事创作</p>
            <p><strong>💡 两种模式：</strong></p>
            <p>• 简单描述：快速生成图片的基本描述（仅 BLIP）</p>
            <p>• 创作故事：根据图片创作完整故事（BLIP + Qwen）</p>
        </div>
    </div>

    <script>
        let selectedFile = null;
        let currentMode = 'describe';
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const generateBtn = document.getElementById('generateBtn');
        const changeImageBtn = document.getElementById('changeImageBtn');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('resultContainer');
        const resultChinese = document.getElementById('resultChinese');
        const resultEnglish = document.getElementById('resultEnglish');
        const modeHint = document.getElementById('modeHint');
        const modeBtns = document.querySelectorAll('.mode-btn');
        
        // 模式切换
        modeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                modeBtns.forEach(b => {
                    b.classList.remove('active');
                    b.style.background = 'white';
                    b.style.color = '#3498db';
                });
                btn.classList.add('active');
                btn.style.background = '#3498db';
                btn.style.color = 'white';
                
                currentMode = btn.dataset.mode;
                if (currentMode === 'describe') {
                    modeHint.textContent = '快速生成图片的基本描述';
                } else {
                    modeHint.textContent = '根据图片创作一个完整的故事（300-500字）';
                }
            });
        });
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(52, 152, 219, 0.2)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'rgba(52, 152, 219, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(52, 152, 219, 0.05)';
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file || !file.type.startsWith('image/')) {
                alert('请选择图片文件！');
                return;
            }
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                uploadArea.style.display = 'none';
                previewContainer.style.display = 'block';
                resultContainer.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        changeImageBtn.addEventListener('click', () => {
            uploadArea.style.display = 'block';
            previewContainer.style.display = 'none';
            resultContainer.style.display = 'none';
            fileInput.value = '';
            selectedFile = null;
        });
        
        generateBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('mode', currentMode);
            
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            generateBtn.disabled = true;
            
            if (currentMode === 'story') {
                document.querySelector('.loading-text').textContent = 'AI 正在创作故事中...';
            } else {
                document.querySelector('.loading-text').textContent = 'AI 正在分析图片...';
            }
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    if (data.mode === 'story') {
                        resultChinese.textContent = data.story;
                        resultEnglish.textContent = '图像描述：' + data.description;
                        document.querySelector('.result-title').textContent = '✨ AI 创作的故事：';
                    } else {
                        resultChinese.textContent = data.description_cn;
                        resultEnglish.textContent = '原文：' + data.description_en;
                        document.querySelector('.result-title').textContent = '✨ 图像描述：';
                    }
                    resultContainer.style.display = 'block';
                } else {
                    alert('生成失败：' + data.error);
                }
            } catch (error) {
                alert('请求失败：' + error.message);
            } finally {
                loading.style.display = 'none';
                generateBtn.disabled = false;
            }
        });
        
        // 飘落动画
        const emojis = ['🖼️', '✨', '🎨', '📝', '💬', '🌟', '💫', '🎭', '📷', '🖌️'];
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 10 + 20) + 'px';
            document.body.appendChild(item);
            setTimeout(() => item.remove(), 7000);
        }
        for(let i = 0; i < 10; i++) { setTimeout(createFallingItem, i * 150); }
        setInterval(createFallingItem, 150);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/static/background')
def background():
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    return '', 404

@app.route('/generate', methods=['POST'])
def generate():
    """生成内容（支持两种模式）"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有上传图片'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'})
        
        # 获取模式
        mode = request.form.get('mode', 'describe')
        
        image = Image.open(file.stream).convert('RGB')
        
        # 步骤 1：使用 BLIP 生成图像描述
        print(f"📖 模式: {mode} - 步骤 1: 生成图像描述...")
        desc_prompt = "Describe this image in detail, including all visible elements, colors, atmosphere, and mood."
        inputs = processor(image, text=desc_prompt, return_tensors="pt").to(device)
        
        outputs = blip_model.generate(
            **inputs,
            max_length=100,
            min_length=30,
            num_beams=5
        )
        
        english_description = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"   英文描述: {english_description}")
        
        # 翻译描述
        chinese_description = english_description
        if translator:
            try:
                result = translator.translate(english_description, src='en', dest='zh-cn')
                chinese_description = result.text
                print(f"   中文描述: {chinese_description}")
            except Exception as e:
                print(f"翻译失败: {e}")
        
        if mode == 'story':
            # 故事模式：步骤 2 使用 Qwen 生成故事
            print("📖 步骤 2: 根据描述生成故事...")
            story = generate_story_with_qwen(chinese_description)
            print(f"   故事生成完成！({len(story)} 字)")
            
            return jsonify({
                'success': True,
                'mode': 'story',
                'story': story,
                'description': chinese_description
            })
        else:
            # 描述模式：只返回描述
            print("   描述模式完成！")
            return jsonify({
                'success': True,
                'mode': 'describe',
                'description_cn': chinese_description,
                'description_en': english_description
            })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("✅ 服务启动成功！")
    print("=" * 70)
    print("📍 访问地址: http://127.0.0.1:5001")
    print("💡 使用说明:")
    print("   1. 在浏览器中打开上述地址")
    print("   2. 上传图片")
    print("   3. 点击'生成故事'按钮")
    print("   4. AI 会自动创作一个完整的故事")
    print("\n🎨 两步法流程:")
    print("   步骤 1: BLIP 分析图片，生成详细描述")
    print("   步骤 2: Qwen 根据描述，创作精彩故事")
    print("=" * 70)
    print("\n按 Ctrl+C 停止服务\n")
    
    app.run(host='127.0.0.1', port=5001, debug=False)
