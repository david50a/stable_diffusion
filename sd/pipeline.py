import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH=512
HEIGHT=512
LATENTS_WIDTH=WIDTH //8
LATENTS_HEIGHT=HEIGHT // 8
def generate(prompt:str, negative_prompt:str, input_image=None,
              strength=0.8, do_cfg=True, cfg_scale=7.5,sample_name='ddpm',
              n_interface_steps=50, models={},seed=None,device=None,idle_device=None,tokenizer=None):
    with torch.no_grad():
        if not (0<=strength<=1.0):
            raise ValueError('strength must be between 0 and 1')
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        generator=torch.Generator(device=device)
        if seed is None:
            generator.manual_seed()
        else:
            generator.manual_seed(seed)
        clip=models['clip']
        clip.to(device)

        if do_cfg:
            # convert prompt into tokens using the tokenizer
            cond_tokens=tokenizer.batch_decode_plus([prompt],paddinr='max_length',max_length=77).input_ids
            # (batch_size,seq_len)
            cond_tokens=torch.tensor(cond_tokens,dtype=torch.long,device=device)
            # (batch_size,seq_len)->(batch_size,seq_len,dim)
            cond_content=clip(cond_tokens)

            uncond_tokens=tokenizer.batch_decode_plus([prompt],paddinr='max_length',max_length=77).input_ids
            uncond_tokens=torch.tensor(uncond_tokens,dtype=torch.long,device=device)
            # (batch_size,seq_len)->(batch_size,seq_len,dim)
            uncond_content=clip(uncond_tokens)