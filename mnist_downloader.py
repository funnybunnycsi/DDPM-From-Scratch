import os
import gzip
import numpy as np
from PIL import Image
from tqdm import tqdm

def download_data():
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
    ]
    
    for url in urls:
        os.system(f'curl -O {url}')
    os.chdir("..")

def extract_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def extract_labels(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

def save_dataset(images, labels, split):
    base_path = os.path.join('data', split)
    
    for i in range(10):
        os.makedirs(os.path.join(base_path, str(i)), exist_ok=True)
    
    for idx, (image, label) in enumerate(tqdm(zip(images, labels), desc=f"Saving {split} images")):
        image_path = os.path.join(base_path, str(label), f'{idx}.png')
        Image.fromarray(image).save(image_path)

def main():
    download_data()

    train_images = extract_images('data/train-images-idx3-ubyte.gz')
    train_labels = extract_labels('data/train-labels-idx1-ubyte.gz')
    save_dataset(train_images, train_labels, 'train')
    
    test_images = extract_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = extract_labels('data/t10k-labels-idx1-ubyte.gz')
    save_dataset(test_images, test_labels, 'test')


if __name__ == "__main__":
    main()