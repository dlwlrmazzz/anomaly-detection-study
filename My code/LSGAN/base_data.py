import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

BATCH_SIZE = 128
NUM_WORKERS = -1

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        self.on_gpu = isinstance(self.args.get('gpus', None), (str, int))
        self.data_dir = 'mnist_img'

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=128, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=-1, help="Number of additional processes to load data."
        )
        return parser

    def prepare_data(self):
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            data_train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = random_split(data_train, [55000, 5000])

        if stage == 'test' or stage is None:
            self.data_val = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)