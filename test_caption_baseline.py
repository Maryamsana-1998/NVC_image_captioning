import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from train_video import RateDistortionLoss
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np
from utils import *
import glob

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

def get_caption(frame):
    prompt = "<image>\nUSER:Give a detailed visual description of this image ?\nASSISTANT:"
    inputs = processor(text=prompt, images=frame, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_length=200)
    caption_llava = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return caption_llava.split('ASSISTANT:')[1]  

def reconstruct_img(caption,size):
    w,h= size
    image = pipeline(caption,  height=h, width=w).images[0]
    return image

img_files = sorted(glob.glob('Jockey/*.png'))
bpp_values =[]
psnr_values= []

for i, file in enumerate(img_files[0:100]):
    frame = Image.open(file)
    size = frame.size
    caption = get_caption(frame)
    r_img = reconstruct_img(caption,size)
    psnr = calculate_psnr(frame, r_img)
    psnr_values.append(psnr)
    temp_file_path = f'temp_caption_{i}.txt'
    with open(temp_file_path, 'w') as file:
        file.write(caption)
    
    # Print the path to the caption file
    print(f'Caption stored at: {temp_file_path}')

    bpp = calculate_bpp(temp_file_path,size)
    bpp_values.append(bpp)
    os.remove(temp_file_path)

print('Bpp:', sum(bpp_values)/len(bpp_values))
print('Psnr: ', sum(psnr_values)/len(psnr_values))