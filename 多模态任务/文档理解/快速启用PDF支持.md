# 快速启用PDF支持

## 当前状态
✅ Python包已安装: `pdf2image`, `python-docx`  
❌ 系统工具未安装: `poppler`

## 方法1: 自动安装（推荐）

### 以管理员身份运行
```bash
# 右键点击 "安装poppler.bat" -> 以管理员身份运行
安装poppler.bat
```

安装完成后**重启命令行窗口**即可使用。

## 方法2: 手动安装

### 步骤1: 下载 Poppler
访问: https://github.com/oschwartz10612/poppler-windows/releases

下载最新的 `Release-xxx.zip` 文件（约30MB）

### 步骤2: 解压安装
1. 解压下载的zip文件
2. 将解压后的文件夹重命名为 `poppler`
3. 移动到 `C:\Program Files\poppler`

### 步骤3: 添加到PATH
1. 右键"此电脑" -> 属性 -> 高级系统设置
2. 点击"环境变量"
3. 在"系统变量"中找到"Path"，点击"编辑"
4. 点击"新建"，添加: `C:\Program Files\poppler\Library\bin`
5. 点击"确定"保存

### 步骤4: 验证安装
重启命令行窗口，运行：
```bash
pdftoppm -v
```

如果显示版本信息，说明安装成功！

## 方法3: 使用conda（如果你用conda）

```bash
conda install -c conda-forge poppler
```

## 测试PDF支持

安装完成后，运行文档理解Web服务：
```bash
python 文档理解Web服务.py
```

启动信息应该显示：
```
✅ PDF文档
✅ Word文档 (.docx)
```

## 如果不想安装poppler

### 临时方案
将PDF文档转换为图片后再上传：
1. 打开PDF
2. 截图或打印为图片
3. 上传图片到Web服务

### 在线转换
使用在线工具将PDF转为图片：
- https://www.ilovepdf.com/pdf_to_jpg
- https://smallpdf.com/pdf-to-jpg

## 常见问题

### Q: 安装后还是提示未安装？
A: 需要重启命令行窗口，或者重启电脑

### Q: 权限不足？
A: 以管理员身份运行安装脚本

### Q: 下载速度慢？
A: 可以使用国内镜像或手动下载后安装

### Q: 只想处理图片，不想装poppler？
A: 完全可以！服务默认支持图片格式，不影响使用

## 支持的格式总结

| 格式 | 需要安装 | 状态 |
|------|---------|------|
| JPG/PNG/BMP | 无 | ✅ 已支持 |
| PDF | poppler | ⚠️ 需安装 |
| Word (.docx) | python-docx | ✅ 已安装 |

## 推荐使用方式

**最简单**: 直接使用图片格式（截图、扫描件）  
**最完整**: 安装poppler，支持所有格式
