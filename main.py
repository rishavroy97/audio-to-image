from diffusers import EulerDiscreteScheduler
from custom_pipeline_stable_diffusion import StableDiffusionPipeline
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from transformers.models.clip.configuration_clip import CLIPTextConfig
import ipdb
from transformers import AutoProcessor, HubertModel
from datasets import load_dataset

torch.set_default_dtype(torch.float16)

# clip_model = CLIPTextTransformer(config=config)

# clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=".")
# clip_tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=".")

# AUDIO_INPUT_CAP = 25000
# audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=".")
# audio_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=".")

gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", cache_dir=".", trust_remote_code=True)

text_string = gigaspeech['train'][0]['text']
audio = gigaspeech['train'][4]['audio']['array'].tolist()

model_id = "stabilityai/stable-diffusion-2"
cache_dir = "./models"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, cache_dir=cache_dir)
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, cache_dir=cache_dir)
pipe = pipe.to("cuda")

# pass audio array into pipeline
image = pipe(audio).images[0]
    
image.save("generated_image.png")
