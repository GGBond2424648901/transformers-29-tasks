#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别 Web 服务 - 语音识别师 🎤
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import pipeline
import base64
import tempfile

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("🎤 语音识别 Web 服务 - 语音识别师")
print("=" * 70)

print("\n🔊 正在加载语音识别模型...")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
print("✅ 语音识别师准备完毕！")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 语音识别 - 语音识别师</title>
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
            font-size: 29px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.75;
        }
        
        @keyframes fall {
            0% {
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.75;
            }
            100% {
                transform: translateY(100vh) rotate(360deg) scale(1.31);
                opacity: 0.25;
            }
        }
        
        .container {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.95) 0%, rgba(56, 142, 60, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(76, 175, 80, 0.5);
            padding: 40px;
            max-width: 1000px;
            margin: 20px auto;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(76, 175, 80, 0.6);
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
            color: #c8e6c9;
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
            border: 3px dashed #4caf50;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: rgba(76, 175, 80, 0.05);
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            background: rgba(76, 175, 80, 0.1);
            border-color: #388e3c;
        }
        
        .upload-icon {
            font-size: 3.5em;
            margin-bottom: 15px;
        }
        
        #fileInput {
            display: none;
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
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(76, 175, 80, 0.4);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .result-container {
            background: linear-gradient(135deg, rgba(200, 230, 201, 0.95) 0%, rgba(165, 214, 167, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #4caf50;
        }
        
        .text-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            border-left: 4px solid #4caf50;
        }
        
        .text-content {
            color: #1b5e20;
            font-size: 1.2em;
            line-height: 1.8;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid rgba(76, 175, 80, 0.1);
            border-left-color: #4caf50;
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
        <h1>🎤 语音识别</h1>
        <p class="subtitle">语音识别师帮你转换语音为文字！</p>
        
        <div class="upload-section">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">🎙️</div>
                <div>点击或拖拽音频文件到这里</div>
                <div style="color: #666; font-size: 0.9em; margin-top: 8px;">支持 MP3、WAV、M4A 等格式</div>
                <input type="file" id="fileInput" accept="audio/*">
            </div>
            
            <div class="button-group" id="buttonGroup" style="display: none;">
                <button class="btn-primary" id="recognizeBtn" onclick="recognizeSpeech()">🔍 识别语音</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div style="color: #388e3c; font-size: 1.1em; font-weight: 600;">AI 正在识别语音...</div>
        </div>
        
        <div class="result-container" id="resultContainer"></div>
    </div>

    <script>
        const fallingItems = ['🎤', '🎙️', '🔊', '🔉', '🔈', '📢', '📣', '🎵', '🎶', '📝', '✨', '⭐', '🌟', '💫', '🎪', '🎭'];
        
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 19 + 23) + 'px';
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
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(76, 175, 80, 0.15)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'rgba(76, 175, 80, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(76, 175, 80, 0.05)';
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file || !file.type.startsWith('audio/')) {
                alert('请选择音频文件！');
                return;
            }
            
            selectedFile = file;
            document.getElementById('buttonGroup').style.display = 'flex';
            document.getElementById('resultContainer').style.display = 'none';
            uploadArea.innerHTML = `
                <div class="upload-icon">✅</div>
                <div>已选择: ${file.name}</div>
                <div style="color: #666; font-size: 0.9em; margin-top: 8px;">点击重新选择</div>
            `;
        }
        
        async function recognizeSpeech() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            const loading = document.getElementById('loading');
            const resultContainer = document.getElementById('resultContainer');
            const recognizeBtn = document.getElementById('recognizeBtn');
            
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            recognizeBtn.disabled = true;
            
            try {
                const response = await fetch('/recognize', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResult(data);
                } else {
                    alert('识别失败：' + data.error);
                }
            } catch (error) {
                alert('请求失败：' + error.message);
            } finally {
                loading.style.display = 'none';
                recognizeBtn.disabled = false;
            }
        }
        
        function displayResult(data) {
            const container = document.getElementById('resultContainer');
            
            let html = '<h3 style="color: #388e3c; margin-bottom: 20px; text-align: center;">✨ 识别结果</h3>';
            
            html += '<div class="text-box">';
            html += '<div style="color: #388e3c; font-weight: bold; margin-bottom: 10px;">📝 识别文本：</div>';
            html += `<div class="text-content">${data.text}</div>`;
            html += '</div>';
            
            container.innerHTML = html;
            container.style.display = 'block';
        }
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

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有上传音频'})
        
        file = request.files['audio']
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # 语音识别
        result = asr(tmp_path)
        text = result['text']
        
        os.unlink(tmp_path)
        
        return jsonify({
            'success': True,
            'text': text
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
    print("🎤 启动语音识别师...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:9003")
    print("🔊 语音识别师在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:9003')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=9003, debug=False)
