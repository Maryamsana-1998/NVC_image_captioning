import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from train_video import RateDistortionLoss
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration,  VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import numpy as np
from utils import *
import glob

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

# model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

vit = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = vit.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds



# def get_caption(frame):
#     prompt = "<image>\nUSER:Give a detailed visual description of this image ?\nASSISTANT:"
#     inputs = processor(text=prompt, images=frame, return_tensors="pt")
#     generate_ids = model.generate(**inputs, max_length=200)
#     caption_llava = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     return caption_llava.split('ASSISTANT:')[1]  

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
    # caption = get_caption(frame)
    caption = predict_step([file])[0]
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