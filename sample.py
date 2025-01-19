import yaml
import os
import argparse
import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

from linear_noise_scheduler import LinearNoiseScheduler
from mnist import MNIST
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    model = UNet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']),
                                     map_location=device))
    model.eval()

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['T'],
                                     beta_start=diffusion_config['beta_1'],
                                     beta_end=diffusion_config['beta_T'])
    
    with torch.inference_mode():
        xt = torch.randn((train_config['num_samples'],
                          model_config['img_channels'],
                          model_config['img_size'],
                          model_config['img_size'])).to(device)
        
        for t in tqdm(reversed(range(diffusion_config["T"]))):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(dim=0).to(device))
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims+1)/2
            grid = make_grid(ims, nrow=train_config["num_grid_rows"])
            img = transforms.ToPILImage()(grid)
            if not os.path.exists(os.path.join(train_config["task_name"], 'samples')):
                os.mkdir(os.path.join(train_config["task_name"], "samples"))
            
            img.save(os.path.join(train_config["task_name"], "samples", f"x0_{t}.png"))
            img.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPM image generation arguments')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    sample(args)