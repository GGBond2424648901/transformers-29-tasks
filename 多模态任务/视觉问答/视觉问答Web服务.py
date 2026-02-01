#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉问答 Web 服务 - 视觉智者 👁️
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import pipeline
from PIL import Image
import base64
import io

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("👁️ 视觉问答 Web 服务 - 视觉智者")
print("=" * 70)

print("\n🔮 正在加载视觉问答模型...")
vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
print("✅ 视觉智者准备完毕！")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>👁️ 视觉问答 - 视觉智者</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
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
            font-size: 27px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.71;
        }
        
        @keyframes fall {
            0% {
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.71;
            }
            100% {
                transform: translateY(100vh) rotate(360deg) scale(1.27);
                opacity: 0.21;
            }
        }
        
        .container {
            background: linear-gradient(135deg, rgba(233, 30, 99, 0.95) 0%, rgba(194, 24, 91, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(233, 30, 99, 0.5);
            padding: 40px;
            max-width: 1000px;
            margin: 20px auto;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(233, 30, 99, 0.6);
            position: relative;
            z-index: 10;
        }
        
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            text-align: center;
            color: #f8bbd0;
            margin-bottom: 30px;
            font-size: 1.2em;
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
        }
        
        .upload-area {
            border: 3px dashed #e91e63;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: rgba(233, 30, 99, 0.05);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            background: rgba(233, 30, 99, 0.1);
            border-color: #c2185b;
        }
        
        .upload-icon {
            font-size: 3.5em;
            margin-bottom: 15px;
        }
        
        #fileInput {
            display: none;
        }
        
        .preview-container {
            display: none;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            display: block;
            margin: 0 auto 20px;
        }
        
        .question-area {
            margin-top: 20px;
        }
        
        .question-area label {
            display: block;
            color: #c2185b;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e91e63;
            border-radius: 15px;
            font-size: 1.05em;
            transition: all 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #c2185b;
            box-shadow: 0 0 15px rgba(233, 30, 99, 0.3);
        }
        
        .quick-questions {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .quick-btn {
            padding: 8px 15px;
            background: linear-gradient(135deg, #ec407a 0%, #e91e63 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        
        .quick-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(233, 30, 99, 0.4);
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        
        button {
            flex: 1;
            padding: 15px;
            font-size: 1.2em;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #e91e63 0%, #c2185b 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(233, 30, 99, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #ec407a 0%, #d81b60 100%);
            color: white;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(236, 64, 122, 0.4);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .result-container {
            background: linear-gradient(135deg, rgba(248, 187, 208, 0.95) 0%, rgba(244, 143, 177, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #e91e63;
        }
        
        .answer-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #e91e63;
        }
        
        .question-text {
            color: #c2185b;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .answer-text {
            color: #880e4f;
            font-size: 1.3em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .confidence {
            color: #666;
            font-size: 0.95em;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #fce4ec;
            border-radius: 4px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #e91e63 0%, #c2185b 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid rgba(233, 30, 99, 0.1);
            border-left-color: #e91e63;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>👁️ 视觉问答</h1>
        <p class="subtitle">视觉智者帮你理解图像内容！</p>
        
        <div class="upload-section">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">🖼️</div>
                <div>点击或拖拽图片到这里</div>
                <div style="color: #666; font-size: 0.9em; margin-top: 8px;">支持 JPG、PNG 等格式</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="preview-container" id="previewContainer">
                <img id="previewImage" class="preview-image" alt="图片预览">
                
                <div class="question-area">
                    <label>❓ 向图片提问（英文）：</label>
                    <input type="text" id="questionInput" placeholder="例如：What is in the image?">
                    <div class="quick-questions">
                        <button class="quick-btn" onclick="setQuestion('What is in the image?')">图中有什么</button>
                        <button class="quick-btn" onclick="setQuestion('What color is it?')">什么颜色</button>
                        <button class="quick-btn" onclick="setQuestion('How many people?')">多少人</button>
                        <button class="quick-btn" onclick="setQuestion('Where is this?')">在哪里</button>
                    </div>
                </div>
                
                <div class="button-group">
                    <button class="btn-primary" id="askBtn" onclick="askQuestion()">🔍 提问</button>
                    <button class="btn-secondary" id="changeBtn" onclick="changeImage()">🔄 更换图片</button>
                </div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div style="color: #c2185b; font-size: 1.1em; font-weight: 600;">AI 正在分析图片...</div>
        </div>
        
        <div class="result-container" id="resultContainer"></div>
    </div>

    <script>
        const fallingItems = ['👁️', '👀', '🔍', '🔎', '🖼️', '📷', '📸', '🎨', '✨', '⭐', '🌟', '💫', '❓', '❔', '💭', '💬'];
        
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 17 + 21) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {
            setTimeout(createFallingItem, i * 150);
        }
        
        setInterval(createFallingItem, 150);
        
        let selectedFile = null;
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(233, 30, 99, 0.15)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'rgba(233, 30, 99, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(233, 30, 99, 0.05)';
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
                document.getElementById('resultContainer').style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        function changeImage() {
            uploadArea.style.display = 'block';
            previewContainer.style.display = 'none';
            document.getElementById('resultContainer').style.display = 'none';
            fileInput.value = '';
            selectedFile = null;
        }
        
        function setQuestion(question) {
            document.getElementById('questionInput').value = question;
        }
        
        async function askQuestion() {
            if (!selectedFile) return;
            
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                alert('请输入问题！');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('question', question);
            
            const loading = document.getElementById('loading');
            const resultContainer = document.getElementById('resultContainer');
            const askBtn = document.getElementById('askBtn');
            
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            askBtn.disabled = true;
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResult(data);
                } else {
                    alert('分析失败：' + data.error);
                }
            } catch (error) {
                alert('请求失败：' + error.message);
            } finally {
                loading.style.display = 'none';
                askBtn.disabled = false;
            }
        }
        
        function displayResult(data) {
            const container = document.getElementById('resultContainer');
            const confidence = (data.score * 100).toFixed(1);
            
            let html = '<h3 style="color: #c2185b; margin-bottom: 20px; text-align: center;">✨ 回答结果</h3>';
            
            html += '<div class="answer-box">';
            html += `<div class="question-text">❓ ${data.question}</div>`;
            html += `<div class="answer-text">💡 ${data.answer}</div>`;
            html += `<div class="confidence">置信度: ${confidence}%</div>`;
            html += '<div class="confidence-bar">';
            html += `<div class="confidence-fill" style="width: ${confidence}%"></div>`;
            html += '</div>';
            html += '</div>';
            
            container.innerHTML = html;
            container.style.display = 'block';
        }
        
        document.getElementById('questionInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
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
    else:
        return '', 404

@app.route('/ask', methods=['POST'])
def ask():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有上传图片'})
        
        file = request.files['image']
        question = request.form.get('question', '')
        
        if not question:
            return jsonify({'success': False, 'error': '请输入问题'})
        
        image = Image.open(file.stream).convert('RGB')
        
        result = vqa(image=image, question=question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result[0]['answer'],
            'score': float(result[0]['score'])
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("👁️ 启动视觉智者...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:8003")
    print("🔍 视觉智者在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:8003')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=8003, debug=False)
