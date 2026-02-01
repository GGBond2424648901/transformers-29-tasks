@echo off
chcp 65001 >nul

echo ============================================================
echo 🚀 启动问答系统 Web 服务
echo ============================================================
echo.

cd /d "%~dp0"

echo 📍 当前目录: %CD%
echo.

echo 🔍 检查 Flask...
D:\aaaalokda\envs\myenv\python.exe -c "import flask" 2>nul
if errorlevel 1 (
    echo ⚠️  Flask 未安装，正在安装...
    D:\aaaalokda\envs\myenv\python.exe -m pip install flask
    echo.
)

echo ✅ 环境检查完成
echo.
echo ============================================================
echo 🌐 启动服务
echo ============================================================
echo.
echo 💡 服务启动后，在浏览器中访问:
echo    http://127.0.0.1:5000
echo.
echo ⚠️  按 Ctrl+C 停止服务
echo.
echo ============================================================
echo.

D:\aaaalokda\envs\myenv\python.exe 问答系统Web服务.py

pause
