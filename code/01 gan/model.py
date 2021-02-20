import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1*28*28),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.generator(z)
        img = img.reshape(img.size(0), -1)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(1*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )  

    def forward(self, img):
        output = self.discriminator(img)
        return output