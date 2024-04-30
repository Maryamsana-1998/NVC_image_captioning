import torch
import datetime
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image
import torch.multiprocessing as mp
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline,DDIMScheduler
import os

def setup(rank, world_size):
    # Configure the setup for each process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ['NCCL_DEBUG'] = 'INFO'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # print('HELLO')

    # Initialize tokenizer and models
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
    unet = pipe.unet.to(device).half()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device).half()

    # Wrap models with DDP
    text_encoder = DDP(text_encoder, device_ids=[rank])
    unet = DDP(unet, device_ids=[rank])
    vae = DDP(vae, device_ids=[rank])

    dataset = TextImageDataset(
        texts=["A rider riding a horse on a track", "A rider moving forward on the track with horse"],
        image_paths=["data/Jockey/img00100.png", "data/Jockey/img00001.png"],
        tokenizer=tokenizer,
        transform=transforms.ToTensor()
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_batch)

    train([text_encoder, unet, vae],scheduler, dataloader, 2, device)

    cleanup()

class TextImageDataset(Dataset):
    def __init__(self, texts, image_paths, tokenizer, transform=None):
        self.texts = texts
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs.input_ids.squeeze(0), image

def collate_batch(batch):
    text_inputs, images = zip(*batch)
    text_inputs_padded = pad_sequence(text_inputs, batch_first=True, padding_value=0)
    images = torch.stack(images, dim=0)
    return text_inputs_padded, images

def train(models,scheduler, dataloader, epochs, device):
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=1e-4)
    for epoch in range(epochs):
        for text_inputs, images in dataloader:
            text_inputs = text_inputs.to(device)
            images = images.to(device)
            with autocast():
                text_features = models[0].module(text_inputs,return_dict=False)[0]
                latents = models[2].module.encode(images).latent_dist.sample()
                # print('l',latents.shape)
                timesteps = scheduler.set_timesteps(2)
                generated_images = models[1].module(latents, encoder_hidden_states=text_features,timestep=scheduler.timesteps,return_dict=False)[0]
                # print(generated_images)
                # print(generated_images[0])
                loss = torch.nn.functional.mse_loss(generated_images, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Rank {dist.get_rank()}, Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=main, args=(rank, world_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
