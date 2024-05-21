import base64
import os
import time
from uuid import uuid4

import modal
from pydantic import BaseModel

GPU_CONFIG = modal.gpu.A10G()
vol = modal.Volume.from_name("marker-pdf-volume", create_if_missing=True)

def download_model_to_image(MODEL_DIR):
    from marker.models import load_all_models
    from marker.settings import Settings
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download("vikp/surya_det2", local_dir=MODEL_DIR)

    model_list = load_all_models()
    return model_list

def download_pdf(url):
    import requests

    response = requests.get(url)
    if response.status_code == 200:
        print("PDF downloaded successfully")
        return response.content
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
        return None

vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(
        "marker-pdf",
        "hf_transfer",
        "opencv-python-headless",
        "pillow",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"TRANSFORMERS_CACHE": "/data/transformers_cache"})
    .env({"HF_HOME": "/data/hf_home"})
)

web_gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install("pillow")
)

app = modal.App("marker-api")

@app.cls(gpu=GPU_CONFIG, image=vllm_image, cpu=16.0, volumes={"/data": vol})
class Model:
    @modal.enter()
    def start_engine(self):
        from marker.models import load_all_models

        print("🥶 cold starting inference")
        start = time.monotonic_ns()
        self.model_list = load_all_models()
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    def parse_pdf_and_return_markdown(self, pdf_bytes, extract_images=False):
        from marker.convert import convert_single_pdf
        from marker.settings import settings
        import cv2
        import torch

        start = time.monotonic_ns()

        settings.EXTRACT_IMAGES = extract_images
        settings.TORCH_DEVICE = "cuda"
        settings.INFERENCE_RAM = 80
        settings.VRAM_PER_TASK = 16
        full_text, images, out_meta = convert_single_pdf(pdf_bytes, self.model_list, batch_multiplier=5)

        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")
        cv2.destroyAllWindows()
        return full_text, images, out_meta

@app.local_entrypoint()
def main():
    model = Model()
    pdf_url = "https://pub-cc8438e664ef4d32a54c800c7c408282.r2.dev/73256500180.pdf"
    
    # Download the PDF and get the bytes
    pdf_bytes = download_pdf(pdf_url)
    
    if pdf_bytes is None:
        print("Failed to download the PDF. Exiting.")
        return
    
    markdown_text, images, out_meta = model.parse_pdf_and_return_markdown.remote(pdf_bytes, True)
    print(f"Count of Images {len(images)}")
    output_file = "parsed_content.md"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(markdown_text)
    image_data = {}
    for i, (filename, image) in enumerate(images.items()):
        image_filepath = f"image_{i+1}.png"
        image.save(image_filepath, "PNG")

        with open(image_filepath, "rb") as f:
            image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_data[f'image_{i+1}'] = image_base64

    print(f"Parsed content saved to {output_file}")
    

class ConvertRequest(BaseModel):
    pdf_chunk: str

@app.function(image=web_gpu_image)
@modal.web_endpoint(method="POST")
def convert(req: ConvertRequest):
    import io
    import base64

    model = Model()
    pdf_bytes = base64.b64decode(req.pdf_chunk)
    markdown_text, images, out_meta = model.parse_pdf_and_return_markdown.remote(pdf_bytes, True)
    base64_images = {}
    for image_name, image in images.items():
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images[image_name] = base64_img
    print(f"Count of Images {len(images)}")
    return {"markdown": markdown_text, "images": base64_images, "metadata": out_meta}