#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图片保存和HTTP链接生成功能
"""

import base64
import os
from dots_ocr.parser_api import extract_text_from_image_base64

def test_image_save():
    """
    测试图片保存功能
    """
    # 创建一个简单的测试图片（1x1像素的PNG）
    # 这是一个最小的PNG图片的base64编码
    test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    print("测试图片保存和HTTP链接生成功能...")
    print(f"测试图片: {test_image_base64[:50]}...")
    
    # 测试保存图片功能
    result = extract_text_from_image_base64(
        test_image_base64, 
        save_image=True, 
        base_url="http://localhost:8000"
    )
    
    print(f"\n处理结果: {result}")
    
    # 检查static/images目录是否创建
    static_dir = os.path.join(os.getcwd(), 'static', 'images')
    if os.path.exists(static_dir):
        print(f"\n✓ static/images目录已创建: {static_dir}")
        
        # 列出保存的图片文件
        image_files = [f for f in os.listdir(static_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if image_files:
            print(f"✓ 发现保存的图片文件: {image_files}")
        else:
            print("✗ 未发现保存的图片文件")
    else:
        print(f"✗ static/images目录未创建: {static_dir}")

if __name__ == "__main__":
    test_image_save()