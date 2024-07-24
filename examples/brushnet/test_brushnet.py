from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os
# choose the base model here
base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
# base_model_path = "runwayml/stable-diffusion-v1-5"

# input brushnet ckpt path
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"

# choose whether using blended operation
blended = False
captions = {
    'Cityline_Family': "Man and women with a child walking close to the shore with city and sea in the background.",
    'Meadow_Family': "Family on the meadow with trees and forest in the background.",
    'NY_Night': "New york city at the night.",
    'City_Vertical': "Asian city.",
    "Tokyo_Guy": "City of tokyo at night.",
    "Beach": "Beach full of people with apartaments in the background."
}

def pred(img_name: str):
    # if "Cityline_Family" in img_name or "Meadow_Family" in img_name:
    #     return
    base_path = "data/_Benchmark_photos/"
    all_masks = [i for i in os.listdir(f"{base_path}/Masks") if i.__contains__(img_name[:-4])]
    image_path = Path(base_path, img_name).__str__()
    caption = captions[img_name[:-4]]
    for mask in all_masks:
        init_image = cv2.imread(image_path)[:,:,::-1]
        mask_image = 1.*(cv2.imread(Path(base_path, "Masks", mask).__str__()).sum(-1)>255)[:,:,np.newaxis]
        init_image = init_image * (1 - mask_image)

        brushnet_conditioning_scale=1.0
        h,w,_ = init_image.shape
        if w<h:
            scale=512/w
        else:
            scale=512/h
        new_h=int(h*scale)
        new_w=int(w*scale)


        init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB").resize((new_w, new_h), Image.BICUBIC)
        mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB").resize((new_w, new_h), Image.NEAREST)

        generator = torch.Generator("cuda").manual_seed(1234)

        image = pipe(
            prompt=caption, 
            image=init_image, 
            mask=mask_image, 
            num_inference_steps=50,
            generator=generator,
            brushnet_conditioning_scale=brushnet_conditioning_scale
        ).images[0]

        image.save(f"out/sd15_brushnet_{mask}")



brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()


all_files = [i for i in os.listdir("data/_Benchmark_photos") if "jpg" in i or ".png" in i]

for f in all_files:
    pred(f)