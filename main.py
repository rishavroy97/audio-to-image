import os

import torch
import torchaudio
from diffusers import EulerDiscreteScheduler

from custom_pipeline_stable_diffusion import StableDiffusionPipeline

torch.set_default_dtype(torch.float16)

audio_folder = os.path.join('data', 'audio')
audio_files = os.listdir(audio_folder)
audio_file = "data/audio/" + audio_files[0]

audio_path = audio_file
waveform, sample_rate = torchaudio.load(audio_path, channels_first=True)

# convert from stereo to mono
audio = torch.mean(waveform, dim=0, keepdim=True).tolist()

model_id = "stabilityai/stable-diffusion-2"
cache_dir = "./models"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16,
                                               cache_dir=cache_dir)
pipe = pipe.to("cuda")

# pass audio array into pipeline
image = pipe(audio).images[0]

image.save("./images/generated_image.jpg")
