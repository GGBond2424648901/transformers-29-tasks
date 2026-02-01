@echo off
chcp 65001 >nul

REM 切换到脚本所在目录
cd /d "%~dp0"

echo ============================================================
echo 测试情感分析 API
echo ============================================================
echo.
echo 当前目录: %CD%
echo.
echo 请确保 API 服务已启动（模型部署示例.py）
echo.
pause

D:\aaaalokda\envs\myenv\python.exe 快速测试API.py

echo.
pause
