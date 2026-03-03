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
prompt='a blue sky with birds and yellow pigs'
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
    negative_prompt=uncond_prompt,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    input_image=image_input,
    strength=strength,
    sample_name=sampler,
    seed=seed,
    n_interface_steps=num_of_interfaces,
    device=device,
    models=models,
    idle_device='cpu',
    tokenizer=tokenizer
)
PIL.Image.fromarray(output_image).save('output.png')