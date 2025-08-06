#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试FastAPI OCR服务
"""

import requests

def test_dots_ocr_api(parse_base64_image=False):
    """测试OCR API"""
    # API端点
    url = "http://localhost:8000/dots-ocr"
    
    # 测试参数
    params = {
        "input_path": "http://10.101.100.13:8111/pdf/WFO_SDK_Overview.pdf",  # 使用demo目录中的测试图片
        "parse_base64_image": parse_base64_image
    }
    
    try:
        mode_desc = "启用图片解析" if parse_base64_image else "保留原始base64图片"
        print(f"正在测试OCR API ({mode_desc})...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            result = response.json()
            print("API调用成功!")
            print(f"状态: {result.get('status')}")
            md_contents = result.get('result', [])
            print(f"markdown内容数量: {len(md_contents)}")
            # 将markdown内容保存到文件
            output_filename = "ocr_result.md"
            with open(output_filename, 'w', encoding='utf-8') as f:
                for i, content in enumerate(md_contents):
                    # 写入页面标注
                    f.write(f"[page-{i+1}]\n")
                    # 写入页面内容
                    if content:
                        f.write(content)
                    else:
                        f.write("(此页无内容)")
                    # 页面间分割线（最后一页不添加）
                    if i < len(md_contents) - 1:
                        f.write("\n\n---\n\n")
                    else:
                        f.write("\n")
            
            print(f"markdown内容已保存到文件: {output_filename}")
            print(f"总共处理了 {len(md_contents)} 页内容")
            return True
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("无法连接到API服务器，请确保服务器正在运行")
        return False
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    # print("=== 测试1: 默认模式（保留base64图片）===")
    # test_dots_ocr_api(parse_base64_image=False)
    
    print("\n=== 测试2: 图片解析模式 ===")
    test_dots_ocr_api(parse_base64_image=True)