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

            # (2,seq_len,dim) = (2,77,768)
            context=torch.cat([cond_content,uncond_content])
        else:
            # convert it into a list of tokens
            tokens=tokenizer.batch_decode_plus([prompt],paddinr='max_length',max_length=77).input_ids
            tokens=torch.tensor(tokens,dtype=torch.long,device=device)
            # (1,77,768)
            context=clip(tokens)
        to_idle(clip)
        if sample_name=='ddpm':
            sampler=DDPMSampler()
            sampler.set_interface_steps(n_interface_steps)
        else:
            raise ValueError(f'Unknown sampler {sample_name}')
        latents_shape=(1,4,LATENTS_HEIGHT,LATENTS_WIDTH)
        if input_image:
            encoder=models['encoder']
            encoder.to(device)
            input_image=input_image.resize((LATENTS_HEIGHT,LATENTS_WIDTH))
            input_image_tensor=np.array(input_image)
            #(height,width,3)
            input_image_tensor=torch.tensor(input_image_tensor,dtype=torch.float32)
            input_image_tensor=rescale(input_image_tensor,(0,255),(-1,1))
            # (height,width,channels) -> (batch_size,height,width,channels)
            input_image_tensor=input_image_tensor.unsqueeze(0)
            # (batch_size,height,width,channels) -> (batch_size,channels,height,width)
            input_image_tensor=input_image_tensor.permute(0,3,1,2)
            encoder_noise=torch.randn(latents_shape,generator=generator,device=device)
            # run the image through the encoder opf the VAE
            latents=encoder(input_image_tensor,encoder_noise)
            sampler.set_strength(strength)
            latents=sampler.add_noise(latents,sampler.timesteps[0])
            to_idle(encoder)
        else:
            # if we are doing text-to-image, start with random noise
            latents=torch.randn(latents_shape,generator=generator,device=device)
        diffusion=models['diffusion']
        diffusion.to(device)
        timesteps=tqdm(diffusion.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1,320)
            time_embedding=get_time_embedding(timestep).to(device)
            # (batch_size, 4, latent_weight,latent_width)
            model_input=latents
            if do_cfg:
                # (batch_size, 4, latent_weight,latent_width) -> # (2*batch_size, 4, latent_weight,latent_width)
                model_input=model_input.repeat(2,1,1,1)
            # model_output: is the predicted noise by the UNET
            model_output=diffusion(model_input,context,time_embedding)
            if do_cfg:
                output_cond,output_uncond=model_output.chunk(2)
                model_output=cfg_scale*(output_cond-output_uncond)+output_uncond

            # remove noise predicted by UNET
            latents=sampler.step(timestep,latents,model_output)
        to_idle(diffusion)
        decoder=models['decoder']
        decoder.to(device)
        images=decoder(latents)
        to_idle(decoder)
        images=rescale(images,(-1,1),(0,255),clamp=True)
        # (batch_size,channels,height,width,) -> (batch_size,height,width,channels)
        images=images.permute(0,2,3,1)
        images=images.to('cpu',torch.uint8).numpy()
        return images[0]

def rescale(x,old_range,new_range,clamp=False):
    old_min,old_max=old_range
    new_min,new_max=new_range
    x-=old_min
    x*=(new_max-new_min)/(old_max-old_min)
    x+=new_min
    if clamp:
        x=x.clamp(old_min,new_max)
    return x

def get_time_embedding(timestep):
    # (160,0
    freqs=torch.pow(10000,-torch.arange(start=0,end=160,dtype=torch.float32)/160)
    # (1,160)
    x=torch.tensor([timestep],dtype=torch.float32)[:,None]*freqs[None]
    #(1,320)
    return torch.cat([torch.cos(x),torch.sin(x)],dim=-1)