import model_loder
import pipeline
import PIL.Image
from transformers import CLIPTokenizer
import torch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer=CLIPTokenizer('../data/vocab.json',merges='../data/merges.txt')
model_file='../data/v1-5-pruned-emaonly.ckpt'
models=model_loder.preload_models_from_standard_weights(model_file,device)
prompt='A hyper-realistic close-up portrait of an elderly seafaring captain, deeply weathered skin with fine wrinkles and pores, silver beard, wearing a heavy wool turtleneck and a brass-buttoned pea coat. Cinematic lighting, dramatic shadows, soft bokeh background of a misty harbor, 8k resolution, highly detailed, shot on 35mm lens.'
uncond_prompt=''
do_cfg=True
cfg_scale=7
image_input=None
image_path=None
strength=0.9
sampler='ddpm'
num_of_interfaces=50
seed=42
output_image=pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    image_input=image_input,
    image_path=image_path,
    strength=strength,
    sampler=sampler,
    seed=seed,
    sample_name=sampler,
    n_interface_steps=num_of_interfaces,
    device=device,
    models=models,
    idle_device='cpu',
    tokenizer=tokenizer
)
PIL.Image.fromarray(output_image).save('output.png')