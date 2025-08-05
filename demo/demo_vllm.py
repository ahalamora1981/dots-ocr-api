import argparse
import json

from transformers.utils.versions import require_version
from PIL import Image
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.model.inference import inference_with_vllm


parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="localhost")
parser.add_argument("--port", type=str, default="8000")
parser.add_argument("--model_name", type=str, default="model")
parser.add_argument("--prompt_mode", type=str, default="prompt_layout_all_en")

args = parser.parse_args()

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    print(args)
    image_path = "demo/demo_image1.jpg"
    prompt = dict_promptmode_to_prompt[args.prompt_mode]
    image = Image.open(image_path)
    response = inference_with_vllm(
        image,
        prompt, 
        ip=args.ip,
        port=args.port,
        temperature=0.1,
        top_p=0.9,
    )
    with open("demo/demo_image1.json", "w") as f:
        json.dump(response, f, indent=4)


if __name__ == "__main__":
    main()
