#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本转语音 Web 服务 - 语音魔法师 🎤
支持中文语音合成
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加ffmpeg到PATH（音频处理需要）
ffmpeg_path = r'D:\transformers训练\transformers-main\预训练模型下载处\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin'
ffmpeg_exe = os.path.join(ffmpeg_path, 'ffmpeg.exe')
if os.path.exists(ffmpeg_exe):
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
    os.environ['FFMPEG_BINARY'] = ffmpeg_exe
    print(f"✅ ffmpeg已配置: {ffmpeg_exe}")
else:
    print(f"⚠️  ffmpeg不存在: {ffmpeg_exe}")

from flask import Flask, request, jsonify, send_file, render_template_string
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile
import torch
import numpy as np
import base64
import io
import uuid

# 导入uroman用于文本预处理
try:
    from uroman import Uroman
    uroman_converter = Uroman()
    UROMAN_AVAILABLE = True
    print("✅ uroman已加载")
except ImportError:
    UROMAN_AVAILABLE = False
    uroman_converter = None
    print("⚠️  uroman未安装,中文支持可能受限")

BACKGROUND_PATH = r'背景.png'

print("=" * 70)
print("🎤 文本转语音 Web 服务 - 语音魔法师")
print("=" * 70)

print("\n🔊 正在加载中文语音合成模型...")
print("📦 模型: suno/bark-small (支持多语言包括中文)")
try:
    # 使用Bark模型，原生支持中文
    from transformers import pipeline as hf_pipeline
    tts_pipeline = hf_pipeline("text-to-speech", model="suno/bark-small")
    print("✅ 中文语音魔法师准备完毕!")
    model_loaded = True
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("\n💡 提示:")
    print("   1. 检查网络连接")
    print("   2. 已设置镜像: https://hf-mirror.com")
    model_loaded = False
    print("   3. 或手动下载模型到缓存目录")
    print("\n⚠️  服务将继续运行,但生成功能将不可用")
    model = None
    tokenizer = None
    model_loaded = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)

# 存储生成的音频文件
generated_files = {}

background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 文本转语音 - 语音魔法师</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            background: url('/static/background') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            padding: 20px;
            overflow-y: auto;
            overflow-x: hidden;
        }
        
        .falling-item {
            position: fixed;
            font-size: 25px;
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
                opacity: 0.3;
            }
        }
        
        .container {
            background: linear-gradient(135deg, rgba(156, 39, 176, 0.95) 0%, rgba(123, 31, 162, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(156, 39, 176, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 800px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(156, 39, 176, 0.6);
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
            color: #f3e5f5;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        
        .warning-box {
            background: rgba(255, 193, 7, 0.2);
            border: 2px solid #ffc107;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
            color: #fff;
            text-align: center;
        }
        
        .input-area {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group label {
            display: block;
            color: #7b1fa2;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #9c27b0;
            border-radius: 15px;
            font-size: 1em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            resize: vertical;
            min-height: 100px;
            transition: all 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #7b1fa2;
            box-shadow: 0 0 15px rgba(156, 39, 176, 0.3);
        }
        
        .hint {
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }
        
        button {
            width: 100%;
            padding: 18px;
            font-size: 1.3em;
            font-weight: bold;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 6px 20px rgba(156, 39, 176, 0.4);
            background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
            color: white;
            margin-bottom: 15px;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(156, 39, 176, 0.5);
        }
        
        button:disabled {
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-container {
            background: linear-gradient(135deg, rgba(243, 229, 245, 0.95) 0%, rgba(225, 190, 231, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #9c27b0;
        }
        
        audio {
            width: 100%;
            margin: 20px 0;
        }
        
        .download-btn {
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
            margin-top: 15px;
        }
        
        .download-btn:hover {
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 文本转语音</h1>
        <p class="subtitle">语音魔法师帮你把文字变成声音!</p>
        
        <div class="warning-box">
            ✅ 当前模型支持中文语音合成<br>
            可以输入中文或英文文本<br>
            <small>首次使用需要下载模型,请确保网络连接正常</small>
        </div>
        
        <div class="input-area">
            <div class="input-group">
                <label>📝 输入文本 (中文/英文):</label>
                <textarea id="inputText" placeholder="请输入要转换为语音的文本...

示例:
你好，欢迎使用文本转语音服务！
今天天气真不错。
人工智能技术正在改变我们的生活。"></textarea>
                <div class="hint">
                    💡 提示: 支持中文和英文文本
                </div>
            </div>
            
            <button id="generateBtn" onclick="generateSpeech()">
                🎤 生成语音
            </button>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        const fallingItems = ['🎤', '🎵', '🎶', '🔊', '🎧', '🎙️', '📢', '🔉', '🎼', '🎹', '✨', '💫'];
        
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 15 + 20) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {
            setTimeout(createFallingItem, i * 150);
        }
        
        setInterval(createFallingItem, 150);
        
        async function generateSpeech() {
            const inputText = document.getElementById('inputText').value.trim();
            
            if (!inputText) {
                alert('请输入要转换的文本!');
                return;
            }
            
            const resultDiv = document.getElementById('result');
            const generateBtn = document.getElementById('generateBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #7b1fa2; font-size: 1.2em;">🎤 语音魔法师正在施法...</p>';
            resultDiv.style.display = 'block';
            generateBtn.disabled = true;
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: inputText })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ ${data.error}</p>`;
                } else {
                    displayResult(data);
                }
            } catch (error) {
                resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ 生成失败: ${error.message}</p>`;
            } finally {
                generateBtn.disabled = false;
            }
        }
        
        function displayResult(data) {
            let html = '<h3 style="color: #7b1fa2; margin-bottom: 20px; text-align: center;">✨ 语音生成成功!</h3>';
            
            html += '<div style="background: white; padding: 20px; border-radius: 15px; margin-bottom: 15px;">';
            html += '<h4 style="color: #7b1fa2; margin-bottom: 10px;">📝 原文:</h4>';
            html += `<p style="color: #333; line-height: 1.6;">${data.text}</p>`;
            html += '</div>';
            
            html += '<div style="background: white; padding: 20px; border-radius: 15px;">';
            html += '<h4 style="color: #7b1fa2; margin-bottom: 10px;">🔊 生成的语音:</h4>';
            html += `<audio controls src="/audio/${data.audio_id}"></audio>`;
            html += `<button class="download-btn" onclick="downloadAudio('${data.audio_id}')">📥 下载音频</button>`;
            html += '</div>';
            
            document.getElementById('result').innerHTML = html;
        }
        
        function downloadAudio(audioId) {
            window.location.href = `/audio/${audioId}?download=1`;
        }
        
        document.getElementById('inputText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                generateSpeech();
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
def get_background():
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    return '', 404

@app.route('/generate', methods=['POST'])
def generate():
    try:
        if not model_loaded or tts_pipeline is None:
            return jsonify({'error': '模型未加载,请检查网络连接后重启服务'}), 500
            
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': '请输入文本'}), 400
        
        # 限制文本长度
        if len(text) > 200:
            text = text[:200]
            
        print(f"生成语音: {text}")
        
        # 使用pipeline生成语音（会自动处理文本预处理）
        speech = tts_pipeline(text)
        
        # 获取音频数据
        audio_data = speech["audio"].squeeze()
        sampling_rate = speech["sampling_rate"]
        
        # 确保音频数据是numpy数组
        if hasattr(audio_data, 'cpu'):
            audio_data = audio_data.cpu().numpy()
        
        print(f"✅ 语音生成成功！采样率: {sampling_rate} Hz")
        
        # 归一化并转换为16位整数
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # 保存到内存
        audio_id = str(uuid.uuid4())
        buffer = io.BytesIO()
        wavfile.write(buffer, sampling_rate, audio_data)
        buffer.seek(0)
        
        generated_files[audio_id] = buffer.getvalue()
        
        print(f"✅ 语音生成成功: {audio_id}")
        
        return jsonify({
            'text': text,
            'audio_id': audio_id
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': f'生成失败: {str(e)}'}), 500

@app.route('/audio/<audio_id>')
def get_audio(audio_id):
    if audio_id not in generated_files:
        return '音频不存在', 404
    
    audio_data = generated_files[audio_id]
    buffer = io.BytesIO(audio_data)
    buffer.seek(0)
    
    download = request.args.get('download', '0') == '1'
    
    return send_file(
        buffer,
        mimetype='audio/wav',
        as_attachment=download,
        download_name=f'speech_{audio_id}.wav'
    )

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🎤 启动语音魔法师...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:9005")
    print("🎵 语音魔法师在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:9005')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=9005, debug=False)
