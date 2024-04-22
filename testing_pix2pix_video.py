import requests
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline
from PIL import Image
import numpy as np
import pickle
import glob
from pathlib import Path
import os
from utils import *



def download(embedding_url, local_filepath):
    r = requests.get(embedding_url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)

captioner_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(captioner_id)
model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    sd_model_ckpt,
    caption_generator=model,
    caption_processor=processor,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

def get_caption(image):
    caption = pipeline.generate_caption(image)
    return caption

def get_reconstructed_image_v2(caption, inv_latents):
    source_embeds = pipeline.get_embeds(caption, batch_size=2)
    pred_img = pipeline(
            caption,
            source_embeds=source_embeds,
            target_embeds=source_embeds,
            num_inference_steps=50,
            cross_attention_guidance_amount=0.15,
            generator=generator,
            latents=inv_latents,
            negative_prompt=caption,
        ).images[0]
    
    return pred_img


img_files = sorted(glob.glob('Jockey/*.png'))
torch.manual_seed(0)

data = []
bpp_values =[]
latent = None  # Initialize latent variable

for i, file in enumerate(img_files[0:100]):
    frame = Image.open(file).convert("RGB").resize((512, 512))
    caption = get_caption(frame)
    
    if i % 5 == 0 or i == 0:
        generator = torch.manual_seed(0)  # Reset seed
        latent = pipeline.invert(caption, image=frame, generator=generator).latents
        output_name = 'compressed_v2/'+ Path(file).stem + '.pkl'
        with open(output_name, 'wb') as f:
            pickle.dump({'caption': caption , 'tensor': latent}, f)
        bpp = calculate_bpp(output_name,(512,512))
        bpp_values.append(bpp)
    else:
        output_name = 'compressed_v2/'+ Path(file).stem + '.pkl'
        with open(output_name, 'wb') as f:
            pickle.dump({'caption': caption }, f)
        bpp = calculate_bpp(output_name,(512,512))
        bpp_values.append(bpp)
        
    data.append([frame, caption, latent])


psnr_values =[]
for i, d in enumerate(data):
    frame,caption,latent = d
    pred = get_reconstructed_image_v2(caption,latent)
    psnr = calculate_psnr(frame,pred)   
    psnr_values.append(psnr)
    pred.save('compressed_v2/'+ 'im00'+str(i) + '.png')
    print('processing complete:' , 'im00'+str(i) + '.png')

print('bpp:', sum(bpp_values)/len(bpp_values))
print('psnr:', sum(psnr_values)/len(psnr_values))
