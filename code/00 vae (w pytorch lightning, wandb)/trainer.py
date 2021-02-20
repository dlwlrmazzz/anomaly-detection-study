import torch
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
import wandb

def loss_function(output_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(output_x, x.view(-1,784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class Model(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        # log hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)

        self.log('train_loss', loss, on_epoch=True) # Log loss

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)

        self.log('val_loss', loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)

        self.log('test_loss', loss, on_epoch=True)

        return loss

    def test_step_end(self, test_step_outputs):
        dummy_input = torch.zeros(784, device=self.config.cuda)
        model_filename = "model_final.onnx"
        torch.onnx.export(self, dummy_input, model_filename)
        wandb.save(model_filename)

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples):
        super().__init__()
        self.val_imgs = val_samples
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        outputs, _ , _ = pl_module(val_imgs)
        reconstructed_img = outputs.view(-1,28,28).unsqueeze(-1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(reconstructed_img,
                 caption='reconstructed image')],
            })