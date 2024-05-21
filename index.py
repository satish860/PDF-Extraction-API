import base64
import os
import time
from typing import List
from fastapi import FastAPI, UploadFile, File, status, HTTPException
import modal



GPU_CONFIG = modal.gpu.T4()

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
        print(f"PDF downloaded successfully")
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
       "opencv-python-headless"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"TRANSFORMERS_CACHE": "/data/transformers_cache"})
    .env({"HF_HOME": "/data/hf_home"}) 
)


@app.cls(
    gpu=GPU_CONFIG,
    image=vllm_image,
    cpu=16.0,
    volumes={"/data": vol}
)
class Model:
    
    @modal.build()
    def build(self):
        self.model_list = download_model_to_image("/data")

    @modal.enter()
    def start_engine(self):
        from marker.models import load_all_models

        print("🥶 cold starting inference")
        start = time.monotonic_ns()
        self.model_list = load_all_models()
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    def parse_pdf_and_return_markdown(self,pdf_url,extract_images = False):
        from marker.convert import convert_single_pdf
        from marker.settings import settings
        import cv2  # Import cv2 here
        import torch

        start = time.monotonic_ns()
        pdf_bytes = download_pdf(url=pdf_url)
        settings.EXTRACT_IMAGES = extract_images
        settings.TORCH_DEVICE = "cuda"
        settings.INFERENCE_RAM = 80
        settings.VRAM_PER_TASK = 16  
        full_text, images, out_meta = convert_single_pdf(pdf_bytes, self.model_list,batch_multiplier=10)
       
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")
        cv2.destroyAllWindows()  # Add this line to release OpenCV resources
        return full_text,images,out_meta


@app.function()
@modal.web_endpoint()
def f():
    return "Hello world!"

    
