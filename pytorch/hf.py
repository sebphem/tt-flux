from huggingface_hub import login

def grab_api_key():
    from configparser import ConfigParser as CFP
    import os
    from  pathlib import Path
    cf = CFP()
    cf.read(Path(os.path.dirname(__file__)) / '..' / 'keys' / 'api_key.ini')
    return cf['keys']['hf']

login(token=grab_api_key())
# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt).images[0]


import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")