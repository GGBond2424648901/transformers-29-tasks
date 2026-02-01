@echo off
chcp 65001 >nul
echo ========================================
echo 安装 Poppler (PDF支持工具)
echo ========================================
echo.

echo 正在下载 Poppler...
echo 下载地址: https://github.com/oschwartz10612/poppler-windows/releases/latest
echo.

:: 创建临时目录
set TEMP_DIR=%TEMP%\poppler_install
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

:: 下载最新版本的 poppler
echo 正在下载 poppler-24.08.0...
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip' -OutFile '%TEMP_DIR%\poppler.zip'}"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 下载失败！
    echo.
    echo 请手动下载并安装：
    echo 1. 访问: https://github.com/oschwartz10612/poppler-windows/releases
    echo 2. 下载最新的 Release-xxx.zip
    echo 3. 解压到 C:\Program Files\poppler
    echo 4. 添加 C:\Program Files\poppler\Library\bin 到系统PATH
    pause
    exit /b 1
)

echo.
echo 正在解压...
powershell -Command "& {Expand-Archive -Path '%TEMP_DIR%\poppler.zip' -DestinationPath '%TEMP_DIR%' -Force}"

:: 安装到 Program Files
set INSTALL_DIR=C:\Program Files\poppler
echo.
echo 正在安装到 %INSTALL_DIR%...

if exist "%INSTALL_DIR%" (
    echo 删除旧版本...
    rmdir /s /q "%INSTALL_DIR%"
)

:: 移动文件
move "%TEMP_DIR%\poppler-24.08.0" "%INSTALL_DIR%"

:: 添加到PATH
echo.
echo 正在添加到系统PATH...
set POPPLER_BIN=%INSTALL_DIR%\Library\bin

:: 使用PowerShell添加到PATH
powershell -Command "& {$path = [Environment]::GetEnvironmentVariable('Path', 'Machine'); if ($path -notlike '*%POPPLER_BIN%*') {[Environment]::SetEnvironmentVariable('Path', $path + ';%POPPLER_BIN%', 'Machine')}}"

:: 清理临时文件
echo.
echo 清理临时文件...
rmdir /s /q "%TEMP_DIR%"

echo.
echo ========================================
echo ✅ Poppler 安装完成！
echo ========================================
echo.
echo 安装位置: %INSTALL_DIR%
echo PATH已添加: %POPPLER_BIN%
echo.
echo ⚠️  重要：请重启命令行窗口或重启电脑使PATH生效
echo.
pause
