# DDPM from Scratch

This project implements a Denoising Diffusion Probabilistic Model (DDPM) inspired by Stable Diffusion, trained on the MNIST dataset. The model uses a UNet architecture for image generation and denoising.

## Project Overview

The DDPM model is designed to generate new images by reversing a gradual noising process. This project includes the following components:

1. **Linear Noise Scheduler**: Manages the noise scheduling for the diffusion process.
2. **MNIST Dataset**: Custom dataset class for loading and preprocessing MNIST images.
3. **UNet Model**: A UNet-based architecture for image denoising and generation.
4. **Training and Sampling Scripts**: Scripts to train the model and generate new images.

## Project Structure

- `linear_noise_scheduler.py`: Implements the linear noise scheduler for the diffusion process.
- `mnist.py`: Custom dataset class for loading MNIST images.
- `mnist_downloader.py`: Script to download and preprocess MNIST data.
- `sample.py`: Script to generate new images using the trained model.
- `train.py`: Script to train the DDPM model.
- `unet.py`: Implements the UNet architecture.
- `unet_test.py`: Tests for the UNet model.
- `config/default.yaml`: Configuration file for training and sampling.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Pillow
- tqdm
- yaml

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ddpm-from-scratch.git
   cd ddpm-from-scratch
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

Download and preprocess the MNIST dataset:
```bash
python mnist_downloader.py
```

### Training

Train the DDPM model using the following command:
```bash
python train.py --config config/default.yaml --val True
```

### Sampling

Generate new images using the trained model:
```bash
python sample.py --config config/default.yaml --from_val False
```

## Blog Post

For an in-depth explanation of the mathematics behind DDPM and this implementation, check out the blog post: [DDPM from Scratch](https://yashwantherukulla.github.io/From-Scratch/DDPM-from-Scratch).

## Acknowledgements

This project is inspired by the work on Stable Diffusion and various implementations of DDPM. Special thanks to the authors of the original papers and the open-source community for their contributions.

Feel free to explore the code and experiment with different configurations and if you want to, feel free to send a PR 😊
Happy coding! 🚀
