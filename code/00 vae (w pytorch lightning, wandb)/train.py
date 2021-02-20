import argparse
import torch
from torch.nn import functional as F
import torch.optim as optim

from model import VAE
from trainer import Model, ImagePredictionLogger
from dataload import MNISTDataModule

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', default='MNIST generative model')
    parser.add_argument('--model', default='VAE')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA training')

    config = parser.parse_args()
    config.cuda = config.cuda and torch.cuda.is_available()

    return config

def main(config):
    # device
    device = torch.device("cuda" if config.cuda else "cpu")  

    # model
    if config.model == 'VAE':
        model = VAE().to(device)
    model = Model(model, config)

    # data
    data = MNISTDataModule(config)
    data.prepare_data()
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    # samples, _ = next(iter(test_loader))
    # samples = samples[:8, :]

    # logger
    wandb_logger = WandbLogger(project=config.project, name=config.model)

    # trainer
    trainer = pl.Trainer(
    logger=wandb_logger,       # W&B integration
    log_every_n_steps=10,      # set the logging frequency
    gpus=-1,                   # use all GPUs
    max_epochs=config.epochs,  # number of epochs
    deterministic=True,        # keep it deterministic
    #callbacks=[ImagePredictionLogger(samples)]
    )

    # fit the model
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()

    # save last model
    if config.save_model:
        torch.save(model.state_dict(), "VAE.pt")
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)