from torchvision import transforms
from torchvision.datasets import MNIST
import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

# def data_loader(config):
#     kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                     transform=transforms.ToTensor()),
#         batch_size=config.batch_size, shuffle=True, **kwargs)
#     valid_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#         batch_size=config.batch_size, shuffle=True, **kwargs)

#     return train_loader, valid_loader

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, config, data_dir='../data'):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        train_loader = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return test_loader