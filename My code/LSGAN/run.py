import argparse
import torch
from torch.nn import functional as F
import torch.optim as optim

from lit_lsgan import Lsgan
from base_data import BaseDataModule

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', default='MNIST generative model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA training')

    data_group = parser.add_argument_group("Data Args")
    BaseDataModule.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    Lsgan.add_to_argparse(model_group)

    config = parser.parse_args()
    config.cuda = config.cuda and torch.cuda.is_available()

    return config

def main(config):
    # device
    # device = torch.device("cuda" if config.cuda else "cpu")  
    wandb_logger = WandbLogger(project=config.project, name='LSGAN')

    dm = BaseDataModule()
    model = Lsgan()
    trainer = pl.Trainer(
    logger=wandb_logger,       # W&B integration
    log_every_n_steps=10,      # set the logging frequency
    gpus=-1,                   # use all GPUs
    max_epochs=config.epochs,  # number of epochs
    deterministic=True,        # keep it deterministic
    #callbacks=[ImagePredictionLogger(samples)]
    )
    trainer.fit(model, dm)
    
    # logger
    wandb.finish()
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)