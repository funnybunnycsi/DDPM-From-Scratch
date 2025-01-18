import yaml
import os
import argparse
import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

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