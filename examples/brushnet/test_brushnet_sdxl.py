from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os


captions = {
    'Cityline_Family': "Man and women with a child walking close to the shore with city and sea in the background.",
    'Meadow_Family': "Family on the meadow with trees and forest in the background.",
    'NY_Night': "New york city at the night.",
    'City_Vertical': "Asian city.",
    "Tokyo_Guy": "City of tokyo at night.",
    "Beach": "Beach next to the sea with apartaments in the background."
}

def pred(img_name: str):

    base_path = "data/_Benchmark_photos/"
    all_masks = [i for i in os.listdir(f"{base_path}/Masks") if i.__contains__(img_name[:-4])]
    image_path = Path(base_path, img_name).__str__()
    caption = captions[img_name[:-4]]
    for mask in all_masks:
        for i in range(0, 2):
            init_image = cv2.imread(image_path)[:,:,::-1]
            mask_image = 1.*(cv2.imread(Path(base_path, "Masks", mask).__str__()).sum(-1)>255)[:,:,np.newaxis]
            init_image = init_image * (1 - mask_image)

            brushnet_conditioning_scale=1.0
            h,w,_ = init_image.shape
            if i == 0:
                if w<h:
                    scale=1024/w
                else:
                    scale=1024/h
                new_h=int(h*scale)
                new_w=int(w*scale)

                init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB").resize((new_w, new_h), Image.BICUBIC)
                mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB").resize((new_w, new_h), Image.NEAREST)

                generator = torch.Generator("cuda").manual_seed(4321)

                image = pipe(
                    prompt=caption, 
                    image=init_image, 
                    mask=mask_image, 
                    num_inference_steps=50,
                    generator=generator,
                    brushnet_conditioning_scale=brushnet_conditioning_scale
                ).images[0]

                image.save(f"out/sdxlv0_brushnet_{mask}")
            else:
                init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB").resize((1024, 1024), Image.BICUBIC)
                mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB").resize((1024, 1024), Image.NEAREST)

                generator = torch.Generator("cuda").manual_seed(4321)

                image = pipe(
                    prompt=caption, 
                    image=init_image, 
                    mask=mask_image, 
                    num_inference_steps=50,
                    generator=generator,
                    brushnet_conditioning_scale=brushnet_conditioning_scale
                ).images[0]

                image.save(f"out/sdxlv0_1024_brushnet_{mask}")


base_model_path = "data/ckpt/juggernautXL_juggernautX"
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v0"

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
)
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

all_files = [i for i in os.listdir("data/_Benchmark_photos") if "jpg" in i or ".png" in i]

for f in all_files:
    pred(f)