# Dots OCR FastAPI 服务

## 概述

本项目已将原有的main函数转换为FastAPI的GET API，提供OCR解析服务。

## 功能特性

- **端点**: `/dots-ocr`
- **方法**: GET
- **输入参数**: 
  - `input_path` (字符串) - 图片或PDF文件路径
  - `parse_base64_image` (布尔值，可选) - 是否解析base64图片内容，默认为False
- **返回**: JSON格式，包含markdown内容列表
- **服务器**: 内置uvicorn服务器，运行在 `http://localhost:8000`
- **图片处理**: 自动识别markdown中的base64图片并提取其中的文字内容
- **外部API**: 集成Qwen2.5 VL API进行图片文字识别

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

#### 方法一：直接运行
```bash
python dots_ocr/parser_api.py
```

#### 方法二：使用启动脚本
```bash
python start_api_server.py
```

### 3. 访问API

服务器启动后，可以通过以下方式访问：

- **API文档**: http://localhost:8000/docs
- **根路径**: http://localhost:8000/
- **OCR端点**: http://localhost:8000/dots-ocr

## API使用示例

### 使用curl

```bash
# 解析图片（默认保留base64图片）
curl "http://localhost:8000/dots-ocr?input_path=demo/demo_image1.jpg"

# 解析图片并提取base64图片中的文字
curl "http://localhost:8000/dots-ocr?input_path=demo/demo_image1.jpg&parse_base64_image=true"

# 解析PDF（默认保留base64图片）
curl "http://localhost:8000/dots-ocr?input_path=demo/demo_pdf1.pdf"

# 解析PDF并提取base64图片中的文字
curl "http://localhost:8000/dots-ocr?input_path=demo/demo_pdf1.pdf&parse_base64_image=true"
```

### 使用Python requests

```python
import requests

# 示例1: 默认模式（保留base64图片）
response = requests.get(
    "http://localhost:8000/dots-ocr",
    params={"input_path": "demo/demo_image1.jpg"}
)

if response.status_code == 200:
    result = response.json()
    print(f"状态: {result['status']}")
    md_contents = result['result']
    print(f"共解析了 {len(md_contents)} 页")
    for i, content in enumerate(md_contents):
        print(f"第{i+1}页内容预览: {content[:200]}...")
else:
    print(f"错误: {response.text}")

# 示例2: 启用图片文字提取
response = requests.get(
    "http://localhost:8000/dots-ocr",
    params={
        "input_path": "demo/demo_image1.jpg",
        "parse_base64_image": True
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"状态: {result['status']}")
    md_contents = result['result']
    print(f"共解析了 {len(md_contents)} 页")
    for i, content in enumerate(md_contents):
        print(f"第{i+1}页内容预览: {content[:200]}...")
else:
    print(f"错误: {response.text}")
```

### 测试API

运行测试脚本：

```bash
python test_api.py
```

## API响应格式

### 成功响应

```json
{
  "status": "success",
  "result": [
    "# 文档标题\n\n这是第一页的markdown内容...",
    "# 第二页\n\n这是第二页的markdown内容..."
  ]
}
```

**说明**: `result` 字段现在直接包含每页的markdown文本内容，而不是文件路径信息。

### 错误响应

```json
{
  "detail": "OCR解析失败: 错误详情"
}
```

## 图片文字提取功能

### 功能说明

当设置 `parse_base64_image=true` 时，如果OCR解析结果的markdown内容中包含base64格式的图片，系统会自动：

1. **识别图片**: 使用正则表达式匹配 `![](data:image;base64,...)` 格式的图片
2. **提取文字**: 调用Qwen2.5 VL API提取图片中的文字内容
3. **内容替换**: 将提取的文字内容替换原图片位置，保持前后换行
4. **格式保持**: 支持表格（markdown格式）和公式（LaTeX格式）的识别

### 外部API配置

- **Qwen2.5 VL服务器**: 10.101.100.11:8028
- **图片接口**: POST /ask-image
- **超时时间**: 30秒
- **温度参数**: 0.1（确保输出稳定性）

### 处理示例

**处理前**:
```markdown
这是一个表格：
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...)
下面是更多内容。
```

**处理后**:
```markdown
这是一个表格：

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |

下面是更多内容。
```

## 配置说明

默认配置参数（可在代码中修改）：

- **服务器地址**: 0.0.0.0:8000
- **VLLM服务器**: 10.101.100.13:8202
- **Qwen2.5 VL服务器**: 10.101.100.11:8028
- **模型名称**: model
- **输出目录**: ./output
- **DPI**: 200
- **线程数**: 16
- **提示模式**: prompt_layout_all_en

## 注意事项

1. 确保VLLM服务器正在运行并可访问（10.101.100.13:8202）
2. 输入文件路径可以是本地路径或HTTP URL
3. 支持的文件格式：PDF和常见图片格式（jpg, png等）
4. 解析结果会保存到output目录中
5. 服务器启动时会自动初始化OCR解析器
6. `parse_base64_image` 参数默认为False，此时返回包含原始base64图片的markdown内容
7. 当设置 `parse_base64_image=true` 时：
   - 确保Qwen2.5 VL服务器正在运行并可访问（10.101.100.11:8028）
   - 图片文字提取功能会增加处理时间，具体取决于图片数量和复杂度
   - 如果Qwen2.5 VL API不可用，图片位置会显示"[图片内容提取失败]"

## 故障排除

1. **模块未找到错误**: 确保已安装所有依赖 `pip install -r requirements.txt`
2. **VLLM连接错误**: 检查VLLM服务器是否正在运行（10.101.100.13:8202）
3. **Qwen2.5 VL连接错误**: 检查Qwen2.5 VL服务器是否正在运行（10.101.100.11:8028）
4. **文件路径错误**: 确保输入文件存在且可访问
5. **端口占用**: 如果8000端口被占用，可以修改代码中的端口号
6. **图片处理超时**: 如果图片较多或较大，可能需要增加超时时间
7. **图片格式不支持**: 确保base64图片格式正确（支持jpg、png等常见格式）