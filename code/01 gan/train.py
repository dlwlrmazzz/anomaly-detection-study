import argparse
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from dataload import get_loaders
from trainer import Trainer

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--device', action='store_true', default='cuda:0',
                        help='CUDA training')
    parser.add_argument('--train-ratio', default=0.8,
                        help='train ratio')
    parser.add_argument('--latent-dim', default=128,
                        help='latent dimension')

    config = parser.parse_args()
    return config

def main(config):

    discriminator = Discriminator()
    generator = Generator(config.latent_dim)
    crit = nn.BCELoss()
    
    trainer = Trainer(
            generator, discriminator,
            crit,
            config)

    train_loader, valid_loader, _ = get_loaders(config)

    trainer.train(train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)