
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor , Normalize, Compose
from torchvision.datasets import MNIST

mnist = MNIST(root="data",
              train=True,
              download=True,
              transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
)

img , label=mnist[0]
torch.min(img), torch.max(img)
len(mnist)

img.shape
img.squeeze().shape

import matplotlib.pyplot as plt
plt.imshow(img.squeeze(), cmap="gray")
plt.show()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

torch.min(denorm(img))

from torch.utils.data import DataLoader
batch_size = 100
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
len(data_loader)

for images, labels in data_loader:
    print(images.shape)
    print(labels.shape)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device


input_size=28*28
hidden_layer=256

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_D=nn.Sequential(
            nn.Linear(input_size, hidden_layer),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_layer, hidden_layer),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_layer, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model_D(x)

model_discriminator=Discriminator()
model_discriminator=model_discriminator.to(device)

model_discriminator


latent_size = 64

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model_G = nn.Sequential(
            nn.Linear(latent_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, input_size),
            nn.Tanh()
        )



    def forward(self, x):
        return self.model_G(x)

generator = Generator(latent_size)
generator = generator.to(device)

z = torch.randn(2, latent_size).to(device)
y = generator(z)
print(y.shape)

gen_images=denorm(y.reshape(-1,28,28))

plt.imshow(gen_images[0].cpu().detach(), cmap="gray")
plt.show()

plt.imshow(gen_images[1].cpu().detach().numpy(), cmap="gray")
plt.show()


d_optimizer=torch.optim.Adam(model_discriminator.parameters(), lr=0.0002)
g_optimizer=torch.optim.Adam(generator.parameters(), lr=0.0002)
loss_function=nn.BCELoss()

def reset_grad():
  d_optimizer.zero_grad()
  g_optimizer.zero_grad()


def train_discriminator(images):
  real_labels=torch.ones(batch_size, 1).to(device)
  fake_labels=torch.zeros(batch_size, 1).to(device)

  real_out=model_discriminator(images)
  real_loss=loss_function(real_out, real_labels)
  real_score=real_out

  z=torch.randn(batch_size, latent_size).to(device)
  fake_images=generator(z)
  fake_out=model_discriminator(fake_images)
  fake_loss=loss_function(fake_out, fake_labels)
  fake_score=fake_out

  loss=real_loss+fake_loss
  reset_grad()

  loss.backward()

  d_optimizer.step()
  return loss , real_loss, fake_loss

def train_generator():

  z=torch.randn(batch_size, latent_size).to(device)
  fake_images=generator(z)
  labels=torch.ones(batch_size, 1).to(device)
  g_loss=loss_function(model_discriminator(fake_images), labels)


  reset_grad()
  g_loss.backward()
  g_optimizer.step()

  return g_loss,fake_images

import os
sample_dir= 'samples'
if not os.path.exists(sample_dir):
  os.makedirs(sample_dir)


from IPython.display import Image
from torchvision.utils import save_image

for images, _ in data_loader:
  images = images.reshape(images.size(0),1,28,28)
  save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'), nrow=10)

Image(os.path.join(sample_dir, 'real_images.png'))

sample_vector = torch.randn(batch_size, latent_size).to(device)

def save_fake_images(index):
  fake_images = generator(sample_vector)
  fake_images = fake_images.reshape(fake_images.size(0),1,28,28)
  fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
  print('Saving', fake_fname)
  save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname),nrow=10)

save_fake_images(0)
Image(os.path.join(sample_dir, 'fake_images-0000.png'))

# %%time
# 
# num_epochs = 300
# total_step = len(data_loader)
# d_losses = []
# g_losses = []
# real_scores = []
# fake_scores = []
# 
# for epoch in range(num_epochs):
#     for i, (images, _) in enumerate(data_loader):
#         images = images.reshape(batch_size, -1).to(device)
# 
#         # Train discriminator and generator
#         d_loss, real_score, fake_score = train_discriminator(images)
#         g_loss, fake_images = train_generator()
# 
#         if (i+1)%200==0:
#           d_losses.append(d_loss.item())
#           g_losses.append(g_loss.item())
#           real_scores.append(real_score.mean().item())
#           fake_scores.append(fake_score.mean().item())
#           print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}',
#                 epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item())
# 
# 
#     # Save fake images generated at the end of each epoch
#     save_fake_images(epoch + 1)
#

