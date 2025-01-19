import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from linear_noise_scheduler import LinearNoiseScheduler
from mnist import MNIST
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    print(config)

    diffusion_config = config["diffusion_params"]
    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    scheduler = LinearNoiseScheduler(T=diffusion_config["T"], beta_1=diffusion_config["beta_1"], beta_T=diffusion_config["beta_T"])
    train_dataset = MNIST(img_path=dataset_config["train_path"])
    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=os.cpu_count())
    if args.val==True:
        val_dataset = MNIST(img_path=dataset_config["val_path"])
        val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=os.cpu_count())
    
    model = UNet(model_config).to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    if os.path.exists(os.path.join(train_config['task_name'], train_config['ckpt_name'])):
        print(f"Loading checkpoint: {train_config['ckpt_name']}")
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']),
                                         map_location=device))
        
    epochs = train_config["epochs"]
    optim = Adam(model.parameters(), lr=train_config["lr"])
    loss_fn = nn.MSELoss()

    mean_train_losses = []
    mean_val_losses = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for data in tqdm(train_loader):
            img, _ = data
            img = img.float().to(device)
            noise = torch.randn_like(img).to(device)
            t = torch.randint(0, diffusion_config["T"], (img.shape[0],)).to(device)

            noisy_img = scheduler.add_noise(img, noise, t)
            noise_pred = model(noisy_img, t)

            loss = loss_fn(noise_pred, noise)
            train_losses.append(loss.item())
            
            loss.backward()
            optim.zero_grad()
            optim.step()

        if args.val==True:
            val_losses = []
            model.eval()
            with torch.inference_mode():
                for data in val_dataset:
                    img, _ = data
                    img = img.float().to(device)
                    noise = torch.randn_like(img).to(device)
                    t = torch.randint(0, diffusion_config["T"], (img.shape[0],)).to(device)

                    noisy_img = scheduler.add_noise(img, noise, t)
                    noise_pred = model(noisy_img, t)

                    loss = loss_fn(noise_pred, noise)
                    val_losses.append(loss.item())

        mean_train_losses.append(np.mean(train_losses))
        mean_val_losses.append(np.mean(val_losses))
        if args.val==True:
            print(f"epoch: {epoch} | train loss: {np.mean(train_losses)} | val loss: {np.mean(val_losses)}")
        else:
            print(f"epoch: {epoch} | train loss: {np.mean(train_losses)}")

        torch.save(model.state_dict(), os.path.join(train_config["task_name"], train_config["ckpt_name"]))
    
    print("Training Successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--config", dest="config_path", default="config/default.yaml", type=str)
    parser.add_argument("--val", dest="val", default=False, type=bool)
    args = parser.parse_args()
    train(args)