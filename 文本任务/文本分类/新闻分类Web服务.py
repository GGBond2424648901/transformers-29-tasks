#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻分类 Web 服务
使用训练好的模型进行新闻分类
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline
import json
import base64

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, 'output', 'news_classifier')
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

print("=" * 70)
print("📰 新闻分类 Web 服务")
print("=" * 70)

# 检查模型是否存在
if not os.path.exists(MODEL_DIR):
    print(f"\n❌ 错误: 模型目录不存在: {MODEL_DIR}")
    print("💡 请先运行 新闻分类训练.py 训练模型")
    exit(1)

# 加载标签映射
label_map_path = os.path.join(MODEL_DIR, 'label_map.json')
with open(label_map_path, 'r', encoding='utf-8') as f:
    LABELS = json.load(f)
    LABELS = {int(k): v for k, v in LABELS.items()}

print(f"\n📂 模型目录: {MODEL_DIR}")
print(f"📋 类别: {', '.join(LABELS.values())}")

# 加载模型
print("\n🤖 加载模型...")
classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=0  # 使用GPU
)
print("✅ 模型加载成功！")

# 创建 Flask 应用
app = Flask(__name__)

# 读取背景图片并转换为 base64
background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')

# HTML 模板
HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📰 新闻分类系统</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            padding: 20px;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        
        .falling-item {{
            position: fixed;
            font-size: 24px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.7;
        }}
        
        @keyframes fall {{
            0% {{
                transform: translateY(-10px) rotate(0deg);
                opacity: 0.7;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg);
                opacity: 0.2;
            }}
        }}
        
        .container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 1000px;
            margin: 20px auto;
            position: relative;
            z-index: 10;
        }}
            max-width: 1200px;
            margin: 20px auto;
            max-width: 800px;
            width: 100%;
            backdrop-filter: blur(10px);
        }}
        
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .categories {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
        }}
        
        .category-tag {{
            background: rgba(255, 255, 255, 0.9);
            color: #667eea;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .input-group {{
            margin-bottom: 25px;
        }}
        
        label {{
            display: block;
            margin-bottom: 10px;
            color: #34495e;
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
            transition: all 0.3s;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .button-group {{
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        button {{
            flex: 1;
            padding: 15px 30px;
            font-size: 1.1em;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .classify-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .classify-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        .classify-btn:active {{
            transform: translateY(0);
        }}
        
        .classify-btn:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }}
        
        .clear-btn {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        
        .clear-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
        }}
        
        .result-container {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px;
            margin-top: 25px;
            display: none;
            animation: slideIn 0.5s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .result-title {{
            font-size: 1.3em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        
        .result-category {{
            font-size: 2em;
            color: #667eea;
            margin: 15px 0;
            font-weight: bold;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        .confidence {{
            text-align: center;
            font-size: 1.2em;
            color: #7f8c8d;
            margin-bottom: 20px;
        }}
        
        .all-probabilities {{
            margin-top: 20px;
        }}
        
        .prob-title {{
            font-size: 1.1em;
            color: #34495e;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        
        .prob-item {{
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            background: white;
            padding: 12px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .prob-label {{
            min-width: 80px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .prob-bar-container {{
            flex: 1;
            height: 25px;
            background: #ecf0f1;
            border-radius: 12px;
            overflow: hidden;
            margin: 0 15px;
        }}
        
        .prob-bar {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            transition: width 0.5s ease-out;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .prob-value {{
            min-width: 60px;
            text-align: right;
            font-weight: bold;
            color: #34495e;
        }}
        
        .loading {{
            text-align: center;
            color: #667eea;
            font-size: 1.2em;
            padding: 20px;
        }}
        
        .error {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            text-align: center;
        }}
        
        .examples {{
            margin-top: 30px;
            padding: 20px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 15px;
        }}
        
        .examples-title {{
            font-size: 1.2em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        
        .example-item {{
            background: white;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 4px solid #667eea;
        }}
        
        .example-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        }}
        
        .example-category {{
            font-weight: bold;
            color: #667eea;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📰 新闻分类系统</h1>
        <p class="subtitle">基于 BERT 的智能新闻分类</p>
        
        <div class="categories">
            <span class="category-tag">🔬 科技</span>
            <span class="category-tag">⚽ 体育</span>
            <span class="category-tag">🎬 娱乐</span>
            <span class="category-tag">💰 财经</span>
            <span class="category-tag">🌍 社会</span>
            <span class="category-tag">🏛️ 政治</span>
        </div>
        
        <div class="input-group">
            <label for="newsText">📝 输入新闻标题或内容：</label>
            <textarea 
                id="newsText" 
                placeholder="例如：华为发布最新5G芯片，性能提升50%"
            ></textarea>
        </div>
        
        <div class="button-group">
            <button class="classify-btn" onclick="classifyNews()">
                🚀 开始分类
            </button>
            <button class="clear-btn" onclick="clearAll()">
                🗑️ 清空
            </button>
        </div>
        
        <div id="result" class="result-container"></div>
        
        <div class="examples">
            <div class="examples-title">💡 示例新闻（点击试试）：</div>
            <div class="example-item" onclick="fillExample(this)">
                <span class="example-category">科技</span>
                <span>OpenAI推出GPT-5模型，多模态能力大幅增强</span>
            </div>
            <div class="example-item" onclick="fillExample(this)">
                <span class="example-category">体育</span>
                <span>中国男篮亚洲杯夺冠，时隔多年重回巅峰</span>
            </div>
            <div class="example-item" onclick="fillExample(this)">
                <span class="example-category">娱乐</span>
                <span>流浪地球2票房突破50亿，创历史新高</span>
            </div>
            <div class="example-item" onclick="fillExample(this)">
                <span class="example-category">财经</span>
                <span>A股三大指数集体上涨，沪指重回3000点</span>
            </div>
            <div class="example-item" onclick="fillExample(this)">
                <span class="example-category">社会</span>
                <span>北京今日最高气温达35度，发布高温橙色预警</span>
            </div>
            <div class="example-item" onclick="fillExample(this)">
                <span class="example-category">政治</span>
                <span>教育部发布双减政策，减轻学生课业负担</span>
            </div>
        </div>
    </div>
    
    <script>
        function fillExample(element) {{
            const text = element.textContent.trim();
            const newsText = text.substring(text.indexOf(' ') + 1);
            document.getElementById('newsText').value = newsText;
        }}
        
        function clearAll() {{
            document.getElementById('newsText').value = '';
            document.getElementById('result').style.display = 'none';
        }}
        
        async function classifyNews() {{
            const text = document.getElementById('newsText').value.trim();
            const resultDiv = document.getElementById('result');
            const classifyBtn = document.querySelector('.classify-btn');
            
            if (!text) {{
                resultDiv.innerHTML = '<div class="error">❌ 请输入新闻内容</div>';
                resultDiv.style.display = 'block';
                return;
            }}
            
            // 显示加载状态
            resultDiv.innerHTML = '<div class="loading">⏳ 正在分类中...</div>';
            resultDiv.style.display = 'block';
            classifyBtn.disabled = true;
            
            try {{
                const response = await fetch('/classify', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ text: text }})
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<div class="error">❌ ${{data.error}}</div>`;
                }} else {{
                    displayResult(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<div class="error">❌ 请求失败: ${{error.message}}</div>`;
            }} finally {{
                classifyBtn.disabled = false;
            }}
        }}
        
        function displayResult(data) {{
            const categoryIcons = {{
                '科技': '🔬',
                '体育': '⚽',
                '娱乐': '🎬',
                '财经': '💰',
                '社会': '🌍',
                '政治': '🏛️'
            }};
            
            const icon = categoryIcons[data.category] || '📰';
            
            let html = `
                <div class="result-title">✨ 分类结果</div>
                <div class="result-category">${{icon}} ${{data.category}}</div>
                <div class="confidence">置信度: ${{(data.confidence * 100).toFixed(2)}}%</div>
            `;
            
            if (data.all_probabilities && data.all_probabilities.length > 0) {{
                html += `
                    <div class="all-probabilities">
                        <div class="prob-title">📊 所有类别概率：</div>
                `;
                
                data.all_probabilities.forEach(item => {{
                    const itemIcon = categoryIcons[item.category] || '📰';
                    const percentage = (item.probability * 100).toFixed(2);
                    html += `
                        <div class="prob-item">
                            <div class="prob-label">${{itemIcon}} ${{item.category}}</div>
                            <div class="prob-bar-container">
                                <div class="prob-bar" style="width: ${{percentage}}%">
                                    ${{percentage >= 15 ? percentage + '%' : ''}}
                                </div>
                            </div>
                            <div class="prob-value">${{percentage}}%</div>
                        </div>
                    `;
                }});
                
                html += '</div>';
            }}
            
            document.getElementById('result').innerHTML = html;
        }}
        
        // 支持回车键提交
        document.getElementById('newsText').addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'Enter') {{
                classifyNews();
            }}
        }});
        
        // 飘落动画
        const emojis = ['📰', '📝', '🗞️', '📄', '📑', '🎯', '🏷️', '✨', '🌟', '💫'];
        
        function createFallingItem() {{
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 10 + 20) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => {{
                item.remove();
            }}, 7000);
        }}
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {{
            setTimeout(createFallingItem, i * 150);
        }}
        
        // 持续创建新元素
        setInterval(createFallingItem, 150);
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
    """主页"""
    return HTML_TEMPLATE

@app.route('/classify', methods=['POST'])
def classify():
    """分类接口"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        # 获取所有类别的概率
        results = classifier(text, top_k=len(LABELS))
        
        # 解析结果
        top_result = results[0]
        label_id = int(top_result['label'].split('_')[-1])
        category = LABELS[label_id]
        confidence = top_result['score']
        
        # 所有类别的概率
        all_probs = []
        for result in results:
            label_id = int(result['label'].split('_')[-1])
            all_probs.append({
                'category': LABELS[label_id],
                'probability': result['score']
            })
        
        return jsonify({
            'category': category,
            'confidence': confidence,
            'all_probabilities': all_probs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌐 启动 Web 服务...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:5002")
    print("💡 按 Ctrl+C 停止服务\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
