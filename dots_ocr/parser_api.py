import os
import json
import time
import re
import requests
import base64
import uuid
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md


PROMPTS = list(dict_promptmode_to_prompt.keys())
BASE_URL = "http://localhost:8000"
# BASE_URL = "http://10.101.100.13:8202"

# Timer Decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} takes {end - start} seconds")
        return result
    return wrapper


def extract_text_from_image_base64(image_base64, save_image=True, base_url=BASE_URL):
    """
    调用Qwen2.5 VL API提取图片中的文字内容
    
    Args:
        image_base64: base64编码的图片字符串（完整的data URL格式）
        save_image: 是否保存图片到本地，默认为True
        base_url: 服务器基础URL，用于生成图片访问链接
    
    Returns:
        提取的文字内容，如果save_image为True，会在内容前面添加图片链接
    """
    try:
        # 提取纯base64数据（去掉data:image/xxx;base64,前缀）
        if image_base64.startswith('data:image'):
            # 提取文件格式
            format_match = re.search(r'data:image/([^;]+);base64,', image_base64)
            image_format = format_match.group(1) if format_match else 'png'
            
            # 提取纯base64数据
            base64_data = image_base64.split(',', 1)[1]
        else:
            # 如果没有前缀，默认为png格式
            base64_data = image_base64
            image_format = 'png'
        
        # 保存图片到本地（如果需要）
        image_url = None
        if save_image:
            try:
                # 创建images目录
                images_dir = os.path.join(os.getcwd(), 'static', 'images')
                os.makedirs(images_dir, exist_ok=True)
                
                # 生成唯一文件名
                image_filename = f"{uuid.uuid4().hex}.{image_format}"
                image_path = os.path.join(images_dir, image_filename)
                
                # 解码并保存图片
                image_data = base64.b64decode(base64_data)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                # 生成HTTP访问链接
                image_url = f"static/images/{image_filename}"
                print(f"图片已保存到: {image_path}")
                print(f"图片访问链接: {image_url}")
                
            except Exception as e:
                print(f"保存图片时发生错误: {str(e)}")
                save_image = False
        
        # Qwen2.5 VL API配置
        api_url = "http://10.101.100.11:8028/ask-image"
        
        # 构建请求数据（使用完整的data URL）
        request_data = {
            "prompt": "执行任务",
            "image_base64": image_base64,
            "system_prompt": (
                "你是一个有视觉能力的AI助手，你的任务是提取图片中的文字内容和实物图像内容。"
                "并输出为markdown格式。请将你的输出内容包装在markdown代码块中，格式如下：```markdown\n你的内容\n```。"
                "你只输出内容，不要做任何说明或注解。"
            ),
            "temperature": 0.1
        }
        
        # 发送请求
        response = requests.post(
            api_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", "")
            
            # 从markdown代码块中提取内容
            markdown_pattern = r'```markdown\s*\n(.*?)\n```'
            match = re.search(markdown_pattern, content, re.DOTALL)
            if match:
                extracted_text = match.group(1).strip()
            else:
                # 如果没有找到markdown代码块，返回原内容
                extracted_text = content
            
            # 如果保存了图片，在识别内容前添加图片链接
            if save_image and image_url:
                final_content = f"![图片]({image_url})\n\n{extracted_text}"
            else:
                final_content = extracted_text
                
            return final_content
        else:
            print(f"API调用失败，状态码: {response.status_code}, 响应: {response.text}")
            return "[图片内容提取失败]"
            
    except Exception as e:
        print(f"提取图片文字时发生错误: {str(e)}")
        return "[图片内容提取失败]"


def process_markdown_with_images(markdown_content, base_url=BASE_URL):
    """
    处理markdown内容中的base64图片，提取图片中的文字并替换
    
    Args:
        markdown_content: 包含base64图片的markdown内容
        base_url: 服务器基础URL，用于生成图片访问链接
    
    Returns:
        处理后的markdown内容
    """
    # 匹配base64图片的正则表达式
    image_pattern = r'!\[.*?\]\(data:image[^;]*;base64,[^)]+\)'
    
    def replace_image(match):
        # 提取完整的图片标记
        full_match = match.group(0)
        
        # 提取base64数据部分
        base64_pattern = r'data:image[^;]*;base64,([^)]+)'
        base64_match = re.search(base64_pattern, full_match)
        
        if base64_match:
            # 获取完整的data URL
            data_url_pattern = r'data:image[^;]*;base64,[^)]+'
            data_url_match = re.search(data_url_pattern, full_match)
            
            if data_url_match:
                image_base64 = data_url_match.group(0)
                print(f"正在处理图片: {image_base64[:50]}...")
                
                # 调用API提取文字（会自动保存图片并生成链接）
                extracted_text = extract_text_from_image_base64(image_base64, save_image=True, base_url=base_url)
                
                # 返回提取的文字内容，保持前后换行
                return f"\n\n{extracted_text}\n\n"
        
        return "\n\n[图片内容提取失败]\n\n"
    
    # 替换所有图片
    processed_content = re.sub(image_pattern, replace_image, markdown_content)
    
    return processed_content


class DotsOCRParser:
    """
    parse image or pdf file
    """
    
    def __init__(self, 
            ip='localhost',
            port=8000,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=64,
            dpi = 200, 
            output_dir="./output", 
            min_pixels=None,
            max_pixels=None,
        ):
        self.dpi = dpi

        # default args for vllm server
        self.ip = ip
        self.port = port
        self.model_name = model_name
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS


    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt, 
            model_name=self.model_name,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: 
            assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: 
            assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        response = self._inference_with_vllm(image, prompt)
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w') as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding='utf-8') as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True) # used for clean output or metric of omnidocbench、olmbench 
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path
        return [result]
        
    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source":"pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        num_thread = min(total_pages, self.num_thread)
        print(f"Parsing PDF with {total_pages} pages using {num_thread} threads...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path
        return results

    def parse_file(self, 
        input_path, 
        output_dir="", 
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False
        ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")
        
        # 创建一个字符串，格式是YYYY-MM-DD-hh-mm-ss
        # current_time = time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())

        print(f"Parsing finished, results saving to {save_dir}")
        with open(os.path.join(output_dir, os.path.basename(filename)+'.jsonl'), 'w', encoding='utf-8') as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results


# FastAPI应用实例
app = FastAPI(title="Dots OCR API", description="OCR服务API", version="1.0.0")

# 添加静态文件服务，用于提供图片访问
static_dir = os.path.join(os.getcwd(), 'static')
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 全局OCR解析器实例
dots_ocr_parser = None

OUTPUT = "./output"
IP = "10.101.100.13"
PORT = 8202
MODEL_NAME = "model"
TEMPERATURE = 0.1
TOP_P = 1.0
MAX_COMPLETION_TOKENS = 16384
NUM_THREAD = 64
DPI = 300

def initialize_parser():
    """初始化OCR解析器"""
    global dots_ocr_parser

    dots_ocr_parser = DotsOCRParser(
        ip=IP,
        port=PORT,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        num_thread=NUM_THREAD,
        dpi=DPI,
        output_dir=OUTPUT, 
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化解析器"""
    initialize_parser()

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {"message": "Dots OCR API", "version": "1.0.0", "docs": "/docs"}

@app.get("/dots-ocr")
async def dots_ocr_api(input_path: str, parse_base64_image: bool = False, base_url: str = BASE_URL):
    """
    OCR解析API
    
    Args:
        input_path: 输入文件路径（图片或PDF）
        parse_base64_image: 是否解析base64图片内容，默认为False
        base_url: 服务器基础URL，用于生成图片访问链接，默认为http://localhost:8000
    
    Returns:
        markdown内容列表
    """
    try:
        if dots_ocr_parser is None:
            raise HTTPException(status_code=500, detail="OCR解析器未初始化")
        
        # 执行OCR解析
        result = dots_ocr_parser.parse_file(
            input_path, 
            prompt_mode="prompt_layout_all_en",
            bbox=None,
        )
        
        # 读取每个md_content_path的实际内容
        md_contents = []
        for item in result:
            if 'md_content_path' in item:
                try:
                    with open(item['md_content_path'], 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    
                    # 根据参数决定是否处理base64图片
                    if parse_base64_image:
                        print("正在处理页面中的图片内容...")
                        processed_content = process_markdown_with_images(md_content, base_url=base_url)
                        md_contents.append(processed_content)
                    else:
                        # 直接返回包含base64图片的原始内容
                        md_contents.append(md_content)
                    
                except Exception as e:
                    print(f"读取markdown文件失败: {item['md_content_path']}, 错误: {str(e)}")
                    md_contents.append("")
            else:
                md_contents.append("")
        
        return JSONResponse(content={"status": "success", "result": md_contents})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR解析失败: {str(e)}")

def main():
    """主函数，启动FastAPI服务器"""
    uvicorn.run("dots_ocr.parser_api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()