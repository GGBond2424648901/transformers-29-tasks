@echo off
chcp 65001 >nul
echo ======================================================================
echo 🤖 启动 ChatGLM 智能客服系统
echo ======================================================================
echo.

cd /d "%~dp0"

echo 📍 当前目录: %CD%
echo.

echo 🐍 使用 Python 环境: D:\aaaalokda\envs\myenv\python.exe
echo.

echo 🚀 启动 Web 服务...
echo.

D:\aaaalokda\envs\myenv\python.exe customer_service_web.py

pause
