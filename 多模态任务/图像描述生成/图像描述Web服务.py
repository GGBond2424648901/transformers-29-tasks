#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像描述生成 Web 服务
上传图片，自动生成文字描述
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
import base64

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
print("🚀 正在启动图像描述生成 Web 服务...")
print("=" * 70)

# 加载模型和处理器
print("📦 加载 BLIP 图像描述模型...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 检测是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✅ 图像描述模型加载成功！(设备: {device})")

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
else:
    print("💡 翻译功能不可用，将仅显示英文描述")

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📸 图像描述生成系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
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
            0% {
                transform: translateY(-10px) rotate(0deg);
                opacity: 0.7;
            }
            100% {
                transform: translateY(100vh) rotate(360deg);
                opacity: 0.2;
            }
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
            padding: 40px;
            max-width: 900px;
            width: 100%;
            backdrop-filter: blur(10px);
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            background: rgba(52, 152, 219, 0.1);
            border-color: #2980b9;
        }
        
        .upload-area.dragover {
            background: rgba(52, 152, 219, 0.2);
            border-color: #2980b9;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 15px;
        }
        
        .upload-text {
            font-size: 1.2em;
            color: #34495e;
            margin-bottom: 10px;
        }
        
        .upload-hint {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        #fileInput {
            display: none;
        }
        
        .preview-container {
            display: none;
            margin-bottom: 30px;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            display: block;
            margin: 0 auto 20px;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 20px;
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
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(245, 87, 108, 0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(79, 172, 254, 0.4);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        
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
        
        .caption-item {
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 15px;
            transition: transform 0.2s ease;
        }
        
        .caption-item:hover {
            transform: translateX(5px);
        }
        
        .caption-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }
        
        .caption-text {
            flex: 1;
            color: #34495e;
            font-size: 1.1em;
            line-height: 1.6;
        }
        
        .caption-english {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
            font-style: italic;
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
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: #667eea;
            font-size: 1.1em;
            font-weight: 600;
        }
        
        .examples {
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid rgba(0,0,0,0.1);
        }
        
        .examples-title {
            font-size: 1.2em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .example-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .example-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .example-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .example-icon {
            font-size: 2em;
            margin-bottom: 8px;
        }
        
        .example-text {
            color: #34495e;
            font-size: 0.9em;
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
        
        .info-box p:last-child {
            margin-bottom: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 图像描述生成系统</h1>
        <p class="subtitle">上传图片，AI 自动生成文字描述</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">🖼️</div>
            <div class="upload-text">点击或拖拽图片到这里</div>
            <div class="upload-hint">支持 JPG、PNG、GIF 等格式</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="preview-container" id="previewContainer">
            <img id="previewImage" class="preview-image" alt="预览图片">
            <div class="button-group">
                <button class="btn-primary" id="generateBtn">🎨 生成描述</button>
                <button class="btn-secondary" id="generateMultiBtn">📝 生成多个描述</button>
                <button class="btn-success" id="changeImageBtn">🔄 更换图片</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">AI 正在分析图片...</div>
        </div>
        
        <div class="result-container" id="resultContainer">
            <div class="result-title">✨ 生成的描述：</div>
            <div id="captionResults"></div>
        </div>
        
        <div class="examples">
            <div class="examples-title">💡 应用场景</div>
            <div class="example-grid">
                <div class="example-item">
                    <div class="example-icon">♿</div>
                    <div class="example-text">无障碍辅助 - 为视障人士描述图像</div>
                </div>
                <div class="example-item">
                    <div class="example-icon">�</div>
                    <div class="example-text">图片 SEO - 自动生成 alt 文本</div>
                </div>
                <div class="example-item">
                    <div class="example-icon">📱</div>
                    <div class="example-text">社交媒体 - 自动生成图片说明</div>
                </div>
                <div class="example-item">
                    <div class="example-icon">📚</div>
                    <div class="example-text">内容管理 - 图片自动标注</div>
                </div>
            </div>
        </div>
        
        <div class="info-box">
            <p><strong>🤖 模型：</strong>Salesforce/blip-image-captioning-base + Google 翻译</p>
            <p><strong>💡 提示：</strong>上传清晰的图片可以获得更准确的描述</p>
            <p><strong>🎯 特点：</strong>自动生成中文描述，同时显示英文原文</p>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const generateBtn = document.getElementById('generateBtn');
        const generateMultiBtn = document.getElementById('generateMultiBtn');
        const changeImageBtn = document.getElementById('changeImageBtn');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('resultContainer');
        const captionResults = document.getElementById('captionResults');
        
        // 点击上传区域
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // 文件选择
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        
        // 处理文件
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
        
        // 更换图片
        changeImageBtn.addEventListener('click', () => {
            uploadArea.style.display = 'block';
            previewContainer.style.display = 'none';
            resultContainer.style.display = 'none';
            fileInput.value = '';
            selectedFile = null;
        });
        
        // 生成单个描述
        generateBtn.addEventListener('click', () => {
            generateCaption(false);
        });
        
        // 生成多个描述
        generateMultiBtn.addEventListener('click', () => {
            generateCaption(true);
        });
        
        // 生成描述
        async function generateCaption(multiple) {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('multiple', multiple);
            
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            generateBtn.disabled = true;
            generateMultiBtn.disabled = true;
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data.captions);
                } else {
                    alert('生成失败：' + data.error);
                }
            } catch (error) {
                alert('请求失败：' + error.message);
            } finally {
                loading.style.display = 'none';
                generateBtn.disabled = false;
                generateMultiBtn.disabled = false;
            }
        }
        
        // 显示结果
        function displayResults(captions) {
            captionResults.innerHTML = '';
            
            captions.forEach((caption, index) => {
                const item = document.createElement('div');
                item.className = 'caption-item';
                item.innerHTML = `
                    <div class="caption-number">${index + 1}</div>
                    <div class="caption-text">
                        <div>${caption.chinese}</div>
                        <div class="caption-english">${caption.english}</div>
                    </div>
                `;
                captionResults.appendChild(item);
            });
            
            resultContainer.style.display = 'block';
        }
        
        // 飘落动画
        const emojis = ['🖼️', '📷', '🎨', '🌄', '🌅', '🏞️', '🎭', '✨', '🌟', '💫'];
        
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 10 + 20) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => {
                item.remove();
            }, 7000);
        }
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {
            setTimeout(createFallingItem, i * 150);
        }
        
        // 持续创建新元素
        setInterval(createFallingItem, 150);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/static/background')
def background():
    """提供背景图片"""
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    else:
        return '', 404

@app.route('/generate', methods=['POST'])
def generate():
    """生成图像描述"""
    try:
        # 获取上传的图片
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有上传图片'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'})
        
        # 读取图片
        image = Image.open(file.stream).convert('RGB')
        
        # 是否生成多个描述
        multiple = request.form.get('multiple', 'false').lower() == 'true'
        
        # 处理图片
        inputs = processor(image, return_tensors="pt").to(device)
        
        # 生成英文描述
        if multiple:
            # 生成多个候选描述
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                num_return_sequences=3
            )
            english_captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
        else:
            # 生成单个描述
            output = model.generate(**inputs, max_length=50)
            english_captions = [processor.decode(output[0], skip_special_tokens=True)]
        
        # 翻译成中文
        captions = []
        for eng_text in english_captions:
            chinese_text = eng_text  # 默认使用英文
            
            if translator:
                try:
                    # 使用 Google 翻译
                    result = translator.translate(eng_text, src='en', dest='zh-cn')
                    chinese_text = result.text
                except Exception as e:
                    print(f"翻译失败: {e}")
                    # 翻译失败则使用英文
            
            captions.append({
                'chinese': chinese_text,
                'english': eng_text
            })
        
        return jsonify({
            'success': True,
            'captions': captions
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
    print("📍 访问地址: http://127.0.0.1:5000")
    print("💡 使用说明:")
    print("   1. 在浏览器中打开上述地址")
    print("   2. 上传或拖拽图片到页面")
    print("   3. 点击按钮生成描述")
    print("   4. 可以选择生成单个或多个候选描述")
    print("\n🎨 功能特点:")
    print("   • 支持拖拽上传图片")
    print("   • 实时预览上传的图片")
    print("   • 生成单个或多个候选描述")
    print("   • 自动翻译成中文（同时显示英文原文）")
    print("   • 美观的界面设计")
    print("=" * 70)
    print("\n按 Ctrl+C 停止服务\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False)
