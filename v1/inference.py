from diffusers import PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
import mediapy as media
import torch
from diffusers import StableDiffusionPipeline
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
    # model = body['model']

    accessId = s3['accessId']
    accessSecret = s3['accessSecret']
    bucketName = s3['bucketName']
    endpointUrl = s3['endpointUrl']

    s3 = boto3.resource('s3',
                        endpoint_url=endpointUrl,
                        aws_access_key_id=accessId,
                        aws_secret_access_key=accessSecret,
                        config=Config(signature_version='s3v4')
                        )

# model_id = "stabilityai/stable-diffusion-2-1-base"
# model_id = "stabilityai/stable-diffusion-2-1"
    model_id = "/content/model"

    scheduler = None
  # scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
  # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
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
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision=model_revision,
        )

    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()

    if model_id.endswith('-base'):
        image_length = 512
    else:
        image_length = 768
    remove_safety = False

    config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(config_dir, exist_ok=True)

    file_path = os.path.join(config_dir, "rclone.conf")
    with open(file_path, "w") as file:
        # Write any content you need to the file
        file.write(
            f"[cloudflare_r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = {accessId}\nsecret_access_key = {accessSecret}\nregion = auto\nendpoint = {endpointUrl}\n\n")

    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negative_prompt,
    ).images

    os.makedirs("/content/output", exist_ok=True)

    filenames = []

    for image in images:
        filename = secrets.token_hex(4)
        image.save(f"/content/output/{filename}.png")
        filenames.append(filename)

    token = secrets.token_hex(12)

    subprocess.call(
        ["rclone", "copy", "/content/output/", f"cloudflare_r2:/{bucketName}/{token}"])

    urls = []

    for filename in filenames:
        PATH = f"{token}/{filename}.png"
        url = s3.meta.client.generate_presigned_url('get_object', Params={
                                                    'Bucket': f'{bucketName}', 'Key': f'{PATH}'}, ExpiresIn=3600)
        urls.append(url)

    return {"output": urls}


runpod.serverless.start({"handler": run})
