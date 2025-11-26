# GAN on MNIST Dataset

This project implements a simple **Generative Adversarial Network (GAN)** using PyTorch to generate handwritten digits similar to the MNIST dataset. It demonstrates the fundamentals of GANs, including building a **Generator** and a **Discriminator**, training them, and generating sample images.

## Project Overview

- **Dataset**: MNIST handwritten digits (28x28 grayscale images)  
- **Framework**: PyTorch  
- **Key Components**:
  - **Discriminator**: Neural network that distinguishes real vs fake images.  
  - **Generator**: Neural network that generates fake images from random noise (latent vectors).  
  - **Loss Function**: Binary Cross Entropy (BCE)  
  - **Optimizers**: Adam for both generator and discriminator  

## Features

- Normalizes MNIST images for GAN training.  
- Visualizes real images from the dataset.  
- Generates fake images after training.  
- Saves generated images in a `samples/` directory.  
- Can be extended for longer training and experimenting with hyperparameters.  

## Code Highlights

1. **Data Preparation**  
   - Loads MNIST dataset using `torchvision.datasets`.  
   - Applies transformations: converts to tensors and normalizes to [-1, 1].  
   - Prepares `DataLoader` for batching and shuffling.  

2. **Discriminator and Generator**  
   - **Discriminator**: Linear layers with LeakyReLU, outputs probability of real/fake.  
   - **Generator**: Linear layers with ReLU, outputs images via Tanh activation.  

3. **Training**  
   - Alternates training of discriminator and generator.  
   - Computes loss using real and fake labels.  
   - Optimizers update network weights.  
   - Saves generated images at regular intervals.  

4. **Visualization**  
   - Displays real and generated images using `matplotlib`.  
   - Saves images in `samples/` for inspection.  

## How to Run

1. Install dependencies:
```bash
pip install torch torchvision matplotlib


Got it! Here’s the updated **How to Run** section with a note about creating a virtual environment in VSCode:

````markdown
## How to Run

1. (Optional) Create a virtual environment (recommended if using VSCode):
```bash
python -m venv venv
````

Activate the environment:

* **Windows:** `venv\Scripts\activate`
* **Linux/Mac:** `source venv/bin/activate`

2. Install dependencies:

```bash
pip install torch torchvision matplotlib
```



