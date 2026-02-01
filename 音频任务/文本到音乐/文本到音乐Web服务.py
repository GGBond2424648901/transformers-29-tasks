#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本到音乐 Web 服务 - 音乐魔法师 🎵
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, send_file
from transformers import pipeline
import scipy.io.wavfile as wavfile
import base64
import tempfile

BACKGROUND_PATH = r'背景.png'

print("=" * 70)
print("🎵 文本到音乐 Web 服务 - 音乐魔法师")
print("=" * 70)

print("\n🎼 正在加载音乐生成模型...")
synthesizer = pipeline("text-to-audio", model="facebook/musicgen-small")
print("✅ 音乐魔法师准备完毕！")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)

background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 文本到音乐 - 音乐魔法师</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            padding: 20px;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        
        .falling-item {{
            position: fixed;
            font-size: 30px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.76;
        }}
        
        @keyframes fall {{
            0% {{
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.76;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg) scale(1.32);
                opacity: 0.26;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(156, 39, 176, 0.95) 0%, rgba(123, 31, 162, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(156, 39, 176, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 900px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(156, 39, 176, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .subtitle {{
            text-align: center;
            color: #e1bee7;
            margin-bottom: 30px;
            font-size: 1.2em;
        }}
        
        .input-area {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
        }}
        
        .input-group {{
            margin-bottom: 20px;
        }}
        
        .input-group label {{
            display: block;
            color: #7b1fa2;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #9c27b0;
            border-radius: 15px;
            font-size: 1.05em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            resize: vertical;
            min-height: 120px;
            transition: all 0.3s;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: #7b1fa2;
            box-shadow: 0 0 15px rgba(156, 39, 176, 0.3);
        }}
        
        .hint {{
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }}
        
        .example-box {{
            background: #f3e5f5;
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
            border-left: 4px solid #9c27b0;
        }}
        
        .quick-prompts {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        
        .quick-btn {{
            padding: 8px 15px;
            background: linear-gradient(135deg, #ab47bc 0%, #9c27b0 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }}
        
        .quick-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(156, 39, 176, 0.4);
        }}
        
        button {{
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
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(156, 39, 176, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(225, 190, 231, 0.95) 0%, rgba(206, 147, 216, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #9c27b0;
        }}
        
        .audio-player {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border-left: 4px solid #9c27b0;
        }}
        
        audio {{
            width: 100%;
            margin-top: 15px;
        }}
        
        .download-btn {{
            margin-top: 15px;
            padding: 12px 30px;
            background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1.1em;
            transition: all 0.3s;
        }}
        
        .download-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(156, 39, 176, 0.4);
        }}
        
        .loading {{
            display: none;
            text-align: center;
            padding: 20px;
        }}
        
        .spinner {{
            border: 4px solid rgba(156, 39, 176, 0.1);
            border-left-color: #9c27b0;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 文本到音乐</h1>
        <p class="subtitle">音乐魔法师帮你创作音乐！</p>
        
        <div class="input-area">
            <div class="input-group">
                <label>📝 描述你想要的音乐（英文）：</label>
                <textarea id="promptInput" placeholder="例如：upbeat electronic dance music with a catchy melody"></textarea>
                <div class="hint">
                    💡 提示：描述音乐的风格、节奏、情绪、乐器等
                </div>
                <div class="example-box">
                    <strong>示例：</strong><br>
                    • calm piano music for meditation<br>
                    • energetic rock music with guitar solo<br>
                    • ambient electronic music for studying
                </div>
                <div class="quick-prompts">
                    <button class="quick-btn" onclick="setPrompt('upbeat electronic dance music')">电子舞曲</button>
                    <button class="quick-btn" onclick="setPrompt('calm piano music for meditation')">冥想钢琴</button>
                    <button class="quick-btn" onclick="setPrompt('energetic rock music')">摇滚音乐</button>
                    <button class="quick-btn" onclick="setPrompt('ambient music for studying')">学习背景</button>
                </div>
            </div>
            
            <button id="generateBtn" onclick="generateMusic()">
                🎼 生成音乐
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div style="color: #7b1fa2; font-size: 1.1em; font-weight: 600;">🎵 音乐魔法师正在创作...</div>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        const fallingItems = ['🎵', '🎶', '🎼', '🎹', '🎸', '🎺', '🎷', '🥁', '🎻', '🎤', '✨', '⭐', '🌟', '💫', '🎪', '🎭'];
        
        function createFallingItem() {{
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 20 + 24) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }}
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {
            setTimeout(createFallingItem, i * 150);
        }
        
        setInterval(createFallingItem, 150);
        
        function setPrompt(prompt) {{
            document.getElementById('promptInput').value = prompt;
        }}
        
        async function generateMusic() {{
            const prompt = document.getElementById('promptInput').value.trim();
            
            if (!prompt) {{
                alert('请输入音乐描述！');
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            const loading = document.getElementById('loading');
            const generateBtn = document.getElementById('generateBtn');
            
            loading.style.display = 'block';
            resultDiv.style.display = 'none';
            generateBtn.disabled = true;
            
            try {{
                const response = await fetch('/generate', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ prompt: prompt }})
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    displayResult(data);
                }} else {{
                    alert('生成失败：' + data.error);
                }}
            }} catch (error) {{
                alert('请求失败：' + error.message);
            }} finally {{
                loading.style.display = 'none';
                generateBtn.disabled = false;
            }}
        }}
        
        function displayResult(data) {{
            const resultDiv = document.getElementById('result');
            
            let html = '<h3 style="color: #7b1fa2; margin-bottom: 20px; text-align: center;">✨ 音乐已生成</h3>';
            
            html += '<div class="audio-player">';
            html += `<div style="color: #7b1fa2; font-weight: bold; margin-bottom: 10px;">🎵 ${{data.prompt}}</div>`;
            html += `<audio controls src="/download/${{data.filename}}"></audio>`;
            html += `<br><button class="download-btn" onclick="window.open('/download/${{data.filename}}', '_blank')">💾 下载音乐</button>`;
            html += '</div>';
            
            resultDiv.innerHTML = html;
            resultDiv.style.display = 'block';
        }}
    </script>
</body>
</html>
"""


@app.route('/static/background')
def background():
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    else:
        return '', 404

@app.route('/')
def index():
    return HTML_TEMPLATE

generated_files = {}

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({{'success': False, 'error': '请输入音乐描述'}}), 400
        
        # 生成音乐
        music = synthesizer(prompt, forward_params={{"max_new_tokens": 256}})
        
        # 保存到临时文件
        import uuid
        filename = f"music_{{uuid.uuid4().hex}}.wav"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        wavfile.write(
            filepath,
            rate=music["sampling_rate"],
            data=music["audio"][0]
        )
        
        generated_files[filename] = filepath
        
        return jsonify({{
            'success': True,
            'prompt': prompt,
            'filename': filename
        }})
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {{error_details}}")
        return jsonify({{'success': False, 'error': str(e)}}), 500

@app.route('/download/<filename>')
def download(filename):
    if filename in generated_files:
        return send_file(generated_files[filename], mimetype='audio/wav', as_attachment=True, download_name=filename)
    return 'File not found', 404

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🎵 启动音乐魔法师...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:9001")
    print("🎼 音乐魔法师在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:9001')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=9001, debug=False)
