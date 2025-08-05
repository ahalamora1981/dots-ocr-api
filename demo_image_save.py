#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示图片保存和HTTP链接生成功能
"""

import base64
import os
from dots_ocr.parser_api import extract_text_from_image_base64, process_markdown_with_images

def create_sample_markdown_with_image():
    """
    创建一个包含base64图片的示例markdown内容
    """
    # 创建一个简单的测试图片（1x1像素的PNG）
    test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    markdown_content = f"""
# 测试文档

这是一个包含图片的测试文档。

## 图片示例

下面是一个图片：

![测试图片]({test_image_base64})

图片后面的文字内容。

## 结束

测试完成。
"""
    
    return markdown_content

def demo_image_processing():
    """
    演示图片处理功能
    """
    print("=== 图片保存和HTTP链接生成功能演示 ===")
    print()
    
    # 1. 测试单个图片处理
    print("1. 测试单个图片处理功能")
    test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    print(f"原始图片数据: {test_image_base64[:50]}...")
    
    # 调用图片处理函数（模拟API调用，实际会失败但能测试保存功能）
    result = extract_text_from_image_base64(
        test_image_base64, 
        save_image=True, 
        base_url="http://localhost:8000"
    )
    
    print(f"处理结果: {result}")
    print()
    
    # 2. 测试markdown内容处理
    print("2. 测试markdown内容处理功能")
    markdown_content = create_sample_markdown_with_image()
    
    print("原始markdown内容:")
    print("---")
    print(markdown_content)
    print("---")
    print()
    
    # 处理markdown内容
    processed_content = process_markdown_with_images(
        markdown_content, 
        base_url="http://localhost:8000"
    )
    
    print("处理后的markdown内容:")
    print("---")
    print(processed_content)
    print("---")
    print()
    
    # 3. 检查保存的文件
    print("3. 检查保存的图片文件")
    static_dir = os.path.join(os.getcwd(), 'static', 'images')
    if os.path.exists(static_dir):
        print(f"✓ static/images目录存在: {static_dir}")
        
        image_files = [f for f in os.listdir(static_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if image_files:
            print(f"✓ 发现 {len(image_files)} 个图片文件:")
            for i, file in enumerate(image_files, 1):
                file_path = os.path.join(static_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  {i}. {file} ({file_size} bytes)")
                print(f"     访问链接: http://localhost:8000/static/images/{file}")
        else:
            print("✗ 未发现图片文件")
    else:
        print(f"✗ static/images目录不存在: {static_dir}")
    
    print()
    print("=== 演示完成 ===")
    print()
    print("使用说明:")
    print("1. 启动API服务器: python start_api_server.py")
    print("2. 调用API: GET /dots-ocr?input_path=xxx&parse_base64_image=true")
    print("3. 图片将自动保存到static/images/目录")
    print("4. 生成的HTTP链接可以在浏览器中访问")

if __name__ == "__main__":
    demo_image_processing()