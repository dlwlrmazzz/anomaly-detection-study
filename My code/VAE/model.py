import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, 32)
        self.fc22 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std) 
        # rand_like
            # Returns a tensor with the same size as input 
            # that is filled with random numbers from a uniform distribution
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
