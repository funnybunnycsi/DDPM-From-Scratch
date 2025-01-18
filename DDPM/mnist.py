import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch import nn
from typing import List, Tuple, Optional
import os

class MNIST(Dataset):
    def __init__(self, img_path: str, img_ext: str = "png") -> None:
        self.img_ext = img_ext
        self.images, self.labels = self.load_imgs(img_path)
        
    def load_imgs(self, img_path: str) -> Tuple[List[str], List[int]]:
        assert os.path.exists(img_path), f"Dataset path {img_path} doesn't exist"

        imgs = []
        labels = []
        try:
            for directory in tqdm(os.listdir(img_path), desc=f"Loading data"):
                dir_path = os.path.join(img_path, directory)
                if os.path.isdir(dir_path):
                    for file in os.listdir(dir_path):
                        if file.endswith(f'.{self.img_ext}'):
                            full_path = os.path.join(dir_path, file)
                            imgs.append(full_path)
                            labels.append(int(directory))
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {str(e)}")
        
        return imgs, labels

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        try:
            img = Image.open(self.images[index]).convert('L')  # Convert to grayscale
            img_tensor = torchvision.transforms.ToTensor()(img)
            return img_tensor, self.labels[index]
        except Exception as e:
            raise RuntimeError(f"Error loading image {self.images[index]}: {str(e)}")
        
def test():
        train_path = "DDPM/data/train"
        test_path = "DDPM/data/test"
        
        try:
            train_dataset = MNIST(img_path=train_path)
            test_dataset = MNIST(img_path=test_path)
            
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Test dataset size: {len(test_dataset)}")
            
            train_img, train_label = train_dataset[0]
            test_img, test_label = test_dataset[0]
            
            assert isinstance(train_img, torch.Tensor), "Train image should be a tensor"
            assert isinstance(test_img, torch.Tensor), "Test image should be a tensor"
            
            print(f"First 5 train labels: {[train_dataset.labels[i] for i in range(5)]}")
            print(f"First 5 test labels: {[test_dataset.labels[i] for i in range(5)]}")
            
            print("Basic dataset testing completed successfully!")
            
        except Exception as e:
            print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    test()