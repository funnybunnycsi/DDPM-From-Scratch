import yaml
import os
import argparse
from tqdm import tqdm
import random

import torch
from torchvision import transforms
from torchvision.utils import make_grid

from mnist import MNIST
from linear_noise_scheduler import LinearNoiseScheduler
from unet import UNet

import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    dataset_config = config["dataset_params"]

    model = UNet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']),
                                     map_location=device))
    model.eval()

    scheduler = LinearNoiseScheduler(T=diffusion_config['T'],
                                     beta_1=diffusion_config['beta_1'],
                                     beta_T=diffusion_config['beta_T'])
    
    with torch.inference_mode():
        if args.from_val==False:
            xt = torch.randn((train_config['num_samples'],
                            model_config['img_channels'],
                            model_config['img_size'],
                            model_config['img_size']))
        
        else:
            mnist = MNIST(img_path=dataset_config["val_path"])
            rand_idx = random.randint(0, mnist.__len__())
            xt, _ = mnist.__getitem__(rand_idx)
            xt = xt.unsqueeze(0).to(device)
        time.sleep(5)


        for t in tqdm(reversed(range(diffusion_config["T"]))):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
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
    parser.add_argument('--config', dest='config_path',default='config/default.yaml', type=str)
    parser.add_argument('--from_val', dest='from_val',action="store_true", default=False)
    args = parser.parse_args()
    sample(args)