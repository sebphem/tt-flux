import torch
from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=torch.bfloat16
)

pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
pipe.to(device)

control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
)

prompt = "A girl in city, 25 years old, cool, futuristic"
image = pipe(
    prompt,
    image=init_image,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    strength=0.7,
    num_inference_steps=2,
    guidance_scale=3.5,
).images[0]
image.save("flux_controlnet_img2img.png")