import torch
from torch import autocast
from diffusers import PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline
import mediapy as media
import subprocess
import runpod
import os
import time
import json
import base64
from PIL import Image
import secrets
import boto3
import botocore
from botocore.exceptions import ClientError
from botocore.client import Config
import random


def run(request):

    body = request['input']
    s3 = request['s3Config']
    prompt = body['prompt']
    height = int(body['height'])
    width = int(body['width'])
    num_inference_steps = int(body['num_inference_steps'])
    guidance_scale = float(body['guidance_scale'])
    num_images_per_prompt = int(body['num_images_per_prompt'])
    negative_prompt = body['negative_prompt']
    model_path = body['model_path']
    bucket_name = body['bucket_name']
    token_path = body['token_path']

    accessId = s3['accessId']
    accessSecret = s3['accessSecret']
    endpointUrl = s3['endpointUrl']

    s3 = boto3.resource('s3',
                        endpoint_url=endpointUrl,
                        aws_access_key_id=accessId,
                        aws_secret_access_key=accessSecret,
                        config=Config(signature_version='s3v4')
                        )

    config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(config_dir, exist_ok=True)

    file_path = os.path.join(config_dir, "rclone.conf")
    with open(file_path, "w") as file:
        # Write any content you need to the file
        file.write(
            f"[cloudflare_r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = {accessId}\nsecret_access_key = {accessSecret}\nregion = auto\nendpoint = {endpointUrl}\n\n")

    subprocess.call(
        ["rclone", "copy", f"cloudflare_r2:{model_path}", "/content/model"])
    model_id = "/content/model"

    # scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  # scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
  # scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  # scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    device = "cuda"

    if model_id.startswith("stabilityai/"):
        model_revision = "fp16"
    else:
        model_revision = None

    if scheduler is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            revision=model_revision,
            safety_checker=None,
        ).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision=model_revision,
            safety_checker=None,
        ).to("cuda")

    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = None

    if model_id.endswith('-base'):
        image_length = 512
    else:
        image_length = 768
    remove_safety = False

    g_cuda = torch.Generator(device='cuda')
    seed = random.randint(1, 10)
    g_cuda.manual_seed(seed)

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            generator=g_cuda
        ).images

    os.makedirs("/content/output", exist_ok=True)

    filenames = []

    for image in images:
        filename = secrets.token_hex(4)
        image.save(f"/content/output/{filename}.png")
        filenames.append(filename)

    subprocess.call(
        ["rclone", "copy", "/content/output/", f"cloudflare_r2:/{bucket_name}/{token_path}"])

    urls = []

    for filename in filenames:
        PATH = f"{token_path}/{filename}.png"
        url = s3.meta.client.generate_presigned_url('get_object', Params={
                                                    'Bucket': f'{bucket_name}', 'Key': f'{PATH}'}, ExpiresIn=3600)
        urls.append(url)

    return {"output": urls}


runpod.serverless.start({"handler": run})
