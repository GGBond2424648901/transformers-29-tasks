#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命名实体识别 Web 服务 - 实体猎人 🎯
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
import base64

BACKGROUND_PATH = r'背景.png'

print("=" * 70)
print("🎯 命名实体识别 Web 服务 - 实体猎人")
print("=" * 70)

print("\n🔍 正在加载实体识别模型...")
ner = pipeline("ner", model="bert-base-chinese", aggregation_strategy="simple")
print("✅ 实体猎人准备完毕！")

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
    <title>🎯 命名实体识别 - 实体猎人</title>
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
            font-size: 27px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.72;
        }}
        
        @keyframes fall {{
            0% {{
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.72;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg) scale(1.28);
                opacity: 0.22;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(103, 58, 183, 0.95) 0%, rgba(81, 45, 168, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(103, 58, 183, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(103, 58, 183, 0.6);
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
            color: #d1c4e9;
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
            color: #512da8;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #673ab7;
            border-radius: 15px;
            font-size: 1.05em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            resize: vertical;
            min-height: 150px;
            transition: all 0.3s;
            line-height: 1.8;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: #512da8;
            box-shadow: 0 0 15px rgba(103, 58, 183, 0.3);
        }}
        
        .hint {{
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }}
        
        .example-box {{
            background: #ede7f6;
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
            border-left: 4px solid #673ab7;
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
            box-shadow: 0 6px 20px rgba(103, 58, 183, 0.4);
            background: linear-gradient(135deg, #673ab7 0%, #512da8 100%);
            color: white;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(103, 58, 183, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(209, 196, 233, 0.95) 0%, rgba(179, 157, 219, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #673ab7;
        }}
        
        .highlighted-text {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            line-height: 2.2;
            font-size: 1.1em;
            border-left: 4px solid #673ab7;
        }}
        
        .entity {{
            padding: 3px 8px;
            border-radius: 5px;
            font-weight: bold;
            margin: 0 2px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .entity:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .entity-PER {{
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
            color: white;
        }}
        
        .entity-LOC {{
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
            color: white;
        }}
        
        .entity-ORG {{
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
            color: white;
        }}
        
        .entity-MISC {{
            background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
            color: white;
        }}
        
        .entity-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .entity-card {{
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }}
        
        .entity-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}
        
        .entity-type {{
            font-weight: bold;
            margin-bottom: 8px;
            padding: 5px 10px;
            border-radius: 8px;
            display: inline-block;
            color: white;
        }}
        
        .type-PER {{
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        }}
        
        .type-LOC {{
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        }}
        
        .type-ORG {{
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        }}
        
        .type-MISC {{
            background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
        }}
        
        .entity-word {{
            color: #512da8;
            font-size: 1.2em;
            font-weight: bold;
            margin: 8px 0;
        }}
        
        .entity-score {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 10px;
            background: white;
            border-radius: 8px;
        }}
        
        .legend-color {{
            width: 30px;
            height: 20px;
            border-radius: 5px;
        }}
        
        .legend-PER {{
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        }}
        
        .legend-LOC {{
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        }}
        
        .legend-ORG {{
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        }}
        
        .legend-MISC {{
            background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
        }}
        
        .legend-PER {{
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        }}
        
        .legend-LOC {{
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        }}
        
        .legend-ORG {{
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        }}
        
        .legend-MISC {{
            background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 命名实体识别</h1>
        <p class="subtitle">实体猎人帮你识别文本中的人名、地名、机构名！</p>
        
        <div class="input-area">
            <div class="input-group">
                <label>📝 输入中文文本：</label>
                <textarea id="inputText" placeholder="请输入中文文本..."></textarea>
                <div class="hint">
                    💡 提示：输入包含人名、地名、机构名等实体的文本
                </div>
                <div class="example-box">
                    <strong>示例：</strong><br>
                    李明在北京大学学习，他来自上海市。周末他经常去故宫博物院参观。
                </div>
            </div>
            
            <button id="recognizeBtn" onclick="recognize()">
                🔍 识别实体
            </button>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        const fallingItems = ['🎯', '🔍', '🔎', '👤', '🏢', '🌍', '🗺️', '📍', '🏛️', '🏙️', '✨', '⭐', '🌟', '💫', '🎪', '🎭'];
        
        function createFallingItem() {{
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 17 + 21) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }}
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {{
            setTimeout(createFallingItem, i * 150);
        }}
        
        setInterval(createFallingItem, 150);
        
        const entityTypeNames = {{
            'PER': '人名',
            'LOC': '地名',
            'ORG': '机构',
            'MISC': '其他'
        }};
        
        const entityIcons = {{
            'PER': '👤',
            'LOC': '📍',
            'ORG': '🏢',
            'MISC': '🏷️'
        }};
        
        async function recognize() {{
            const inputText = document.getElementById('inputText').value.trim();
            
            if (!inputText) {{
                alert('请输入文本！');
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            const recognizeBtn = document.getElementById('recognizeBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #512da8; font-size: 1.2em;">🔍 实体猎人正在搜索...</p>';
            resultDiv.style.display = 'block';
            recognizeBtn.disabled = true;
            
            try {{
                const response = await fetch('/recognize', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ text: inputText }})
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResult(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ 识别失败: ${{error.message}}</p>`;
            }} finally {{
                recognizeBtn.disabled = false;
            }}
        }}
        
        function displayResult(data) {{
            let html = '<h3 style="color: #512da8; margin-bottom: 20px; text-align: center;">✨ 识别结果</h3>';
            
            // 图例 - 颜色与高亮文本完全对应
            html += '<div class="legend">';
            html += '<div class="legend-item"><div class="legend-color legend-PER"></div><span>👤 人名</span></div>';
            html += '<div class="legend-item"><div class="legend-color legend-LOC"></div><span>📍 地名</span></div>';
            html += '<div class="legend-item"><div class="legend-color legend-ORG"></div><span>🏢 机构</span></div>';
            html += '<div class="legend-item"><div class="legend-color legend-MISC"></div><span>🏷️ 其他</span></div>';
            html += '</div>';
            
            // 高亮文本
            html += '<div class="highlighted-text">';
            html += data.highlighted_text;
            html += '</div>';
            
            // 实体列表
            if (data.entities.length > 0) {{
                html += '<h4 style="color: #512da8; margin-bottom: 15px;">📋 识别到的实体：</h4>';
                html += '<div class="entity-list">';
                
                data.entities.forEach(entity => {{
                    const typeName = entityTypeNames[entity.entity_group] || entity.entity_group;
                    const icon = entityIcons[entity.entity_group] || '🏷️';
                    const score = (entity.score * 100).toFixed(1);
                    
                    html += `
                        <div class="entity-card">
                            <div class="entity-type type-${{entity.entity_group}}">${{icon}} ${{typeName}}</div>
                            <div class="entity-word">${{entity.word}}</div>
                            <div class="entity-score">置信度: ${{score}}%</div>
                        </div>
                    `;
                }});
                
                html += '</div>';
            }} else {{
                html += '<p style="text-align: center; color: #666; margin-top: 20px;">未识别到命名实体</p>';
            }}
            
            document.getElementById('result').innerHTML = html;
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

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '请输入文本'}), 400
        
        # 识别实体
        entities = ner(text)
        
        # 转换numpy类型为Python原生类型
        for entity in entities:
            entity['score'] = float(entity['score'])
            entity['start'] = int(entity['start'])
            entity['end'] = int(entity['end'])
        
        # 生成高亮文本
        highlighted_text = text
        # 按位置倒序排序，避免替换时位置偏移
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        for entity in sorted_entities:
            word = entity['word']
            entity_type = entity['entity_group']
            start = entity['start']
            end = entity['end']
            
            highlighted = f'<span class="entity entity-{entity_type}" title="{entity_type}: {(entity["score"]*100):.1f}%">{word}</span>'
            highlighted_text = highlighted_text[:start] + highlighted + highlighted_text[end:]
        
        return jsonify({
            'original': text,
            'entities': entities,
            'highlighted_text': highlighted_text
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🎯 启动实体猎人...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:7005")
    print("🔍 实体猎人在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:7005')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=7005, debug=False)
