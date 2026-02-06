#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器翻译 Web 服务 - 翻译精灵 🌍 (演示版)
由于 Windows 系统对某些翻译模型的 sentencepiece tokenizer 支持有限,
本版本使用简单的演示功能
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from flask import Flask, request, jsonify, render_template_string, send_file
import base64

BACKGROUND_PATH = r'背景.png'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)

# 简单的演示翻译字典
DEMO_TRANSLATIONS = {
    "hello": "你好",
    "how are you": "你好吗",
    "good morning": "早上好",
    "good night": "晚安",
    "thank you": "谢谢",
    "goodbye": "再见",
    "yes": "是的",
    "no": "不",
    "please": "请",
    "sorry": "对不起",
    "i love you": "我爱你",
    "welcome": "欢迎",
}

def simple_translate(text):
    """简单的演示翻译功能"""
    text_lower = text.lower().strip()
    
    # 检查是否在演示字典中
    if text_lower in DEMO_TRANSLATIONS:
        return DEMO_TRANSLATIONS[text_lower]
    
    # 简单的单词翻译
    words = text_lower.split()
    translated_words = [DEMO_TRANSLATIONS.get(word, f"[{word}]") for word in words]
    
    return " ".join(translated_words) + " (演示翻译)"

# 读取背景图片
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
    <title>🌍 机器翻译 - 翻译精灵 (演示版)</title>
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
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.95) 0%, rgba(21, 101, 192, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(33, 150, 243, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 900px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(33, 150, 243, 0.6);
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
            color: #e3f2fd;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        
        .demo-notice {
            background: rgba(255, 193, 7, 0.2);
            border: 2px solid #ffc107;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
            color: #fff;
            text-align: center;
        }
        
        .translation-area {
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
            color: #1976d2;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #2196f3;
            border-radius: 15px;
            font-size: 1em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            resize: vertical;
            min-height: 120px;
            transition: all 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #1976d2;
            box-shadow: 0 0 15px rgba(33, 150, 243, 0.3);
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
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
            color: white;
            margin-bottom: 15px;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.5);
        }
        
        button:disabled {
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-container {
            background: linear-gradient(135deg, rgba(227, 242, 253, 0.95) 0%, rgba(187, 222, 251, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #2196f3;
        }
        
        .result-text {
            background: white;
            padding: 20px;
            border-radius: 15px;
            font-size: 1.1em;
            line-height: 1.8;
            color: #333;
            border-left: 4px solid #2196f3;
        }
        
        .language-indicator {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin: 20px 0;
        }
        
        .language-box {
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .arrow {
            font-size: 2em;
            color: #2196f3;
        }
        
        .demo-examples {
            background: rgba(33, 150, 243, 0.15);
            border-radius: 15px;
            padding: 15px;
            margin-top: 15px;
            border: 2px solid rgba(33, 150, 243, 0.3);
        }
        
        .demo-examples h4 {
            color: #1976d2;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .demo-examples ul {
            list-style: none;
            color: #333;
        }
        
        .demo-examples li {
            padding: 5px 0;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 机器翻译</h1>
        <p class="subtitle">翻译精灵帮你跨越语言障碍!</p>
        
        <div class="demo-notice">
            ⚠️ 演示版本 - 由于 Windows 系统限制,当前使用简化翻译功能<br>
            完整版本需要在 Linux 环境或使用其他翻译模型
        </div>
        
        <div class="translation-area">
            <div class="input-group">
                <label>📝 输入英文文本:</label>
                <textarea id="inputText" placeholder="请输入要翻译的英文文本...
例如: Hello, how are you?"></textarea>
            </div>
            
            <div class="language-indicator">
                <div class="language-box">🇬🇧 English</div>
                <div class="arrow">→</div>
                <div class="language-box">🇨🇳 中文</div>
            </div>
            
            <button id="translateBtn" onclick="doTranslate()">
                🌐 开始翻译
            </button>
            
            <div class="demo-examples">
                <h4>💡 支持的演示短语:</h4>
                <ul>
                    <li>• Hello → 你好</li>
                    <li>• How are you → 你好吗</li>
                    <li>• Good morning → 早上好</li>
                    <li>• Thank you → 谢谢</li>
                    <li>• Goodbye → 再见</li>
                </ul>
            </div>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        // 创建飘落的元素
        const fallingItems = ['🌍', '🌎', '🌏', '🗣️', 'A', 'B', 'C', '中', '文', '英', '💬', '📖', '✨', '🌐', '🔤'];
        
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
        
        async function doTranslate() {
            const inputText = document.getElementById('inputText').value.trim();
            
            if (!inputText) {
                alert('请输入要翻译的文本!');
                return;
            }
            
            const resultDiv = document.getElementById('result');
            const translateBtn = document.getElementById('translateBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #1976d2; font-size: 1.2em;">🌐 翻译精灵正在工作...</p>';
            resultDiv.style.display = 'block';
            translateBtn.disabled = true;
            
            try {
                const response = await fetch('/translate', {
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
                resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ 翻译失败: ${error.message}</p>`;
            } finally {
                translateBtn.disabled = false;
            }
        }
        
        function displayResult(data) {
            let html = '<h3 style="color: #1976d2; margin-bottom: 20px; text-align: center;">✨ 翻译结果</h3>';
            
            html += '<div style="margin-bottom: 20px;">';
            html += '<h4 style="color: #1976d2; margin-bottom: 10px;">🇬🇧 原文:</h4>';
            html += `<div class="result-text">${data.original}</div>`;
            html += '</div>';
            
            html += '<div>';
            html += '<h4 style="color: #1976d2; margin-bottom: 10px;">🇨🇳 译文:</h4>';
            html += `<div class="result-text">${data.translation}</div>`;
            html += '</div>';
            
            document.getElementById('result').innerHTML = html;
        }
        
        // 回车键翻译
        document.getElementById('inputText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                doTranslate();
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
    """提供背景图片"""
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    return '', 404

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '请输入要翻译的文本'}), 400
        
        # 使用简单翻译
        translation = simple_translate(text)
        
        return jsonify({
            'original': text,
            'translation': translation
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🌍 机器翻译 Web 服务 - 翻译精灵 (演示版)")
    print("=" * 70)
    print("\n⚠️  注意: 这是演示版本")
    print("💡 由于 Windows 系统对某些翻译模型的 tokenizer 支持有限")
    print("💡 当前使用简化的演示翻译功能")
    print("💡 完整功能建议在 Linux 环境下运行或使用其他翻译API\n")
    print("✅ 翻译精灵准备完毕!")
    
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🌍 启动翻译精灵...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:7001")
    print("🌐 翻译精灵在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:7001')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=7001, debug=False)
