#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动FastAPI OCR服务器
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dots_ocr.parser_api import main

if __name__ == "__main__":
    print("正在启动Dots OCR FastAPI服务器...")
    print("服务器将在 http://localhost:8000 启动")
    print("API端点: http://localhost:8000/dots-ocr")
    print("API文档: http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器时发生错误: {str(e)}")
        sys.exit(1)