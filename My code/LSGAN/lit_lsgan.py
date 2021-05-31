import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import LSGAN

OPTIMIZER = "Adam"
LR = 1e-3
# LOSS = "cross_entropy"

class Lsgan(pl.LightningModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        # self.train_acc = pl.metrics.Accuracy()
        # self.val_acc = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()

        self.generator = LSGAN.Generator()
        self.discriminator = LSGAN.Discriminator()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        # parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        lr = self.args.get("lr", LR)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):  # pylint: disable=unused-argument
        imgs, _ = batch
        # sample noise
        z = torch.randn(1, 100)
        z = z.type_as(imgs)
        # train generator
        if optimizer_idx == 0:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = self.adversarial_loss(
                        self.discriminator(self.generator(z)), valid
            )
            self.log("g_loss", g_loss, on_epoch=True)
            return g_loss

        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(
                            self.discriminator(imgs), valid
            )

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self.generator(z).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss, on_epoch=True)
            return d_loss

    def validation_step(self, batch, batch_idx, optimizer_idx):  # pylint: disable=unused-argument
        self.train_step(batch, batch_idx, optimizer_idx)
        

    def test_step(self, batch, batch_idx, optimizer_idx):  # pylint: disable=unused-argument
        self.train_step(batch, batch_idx, optimizer_idx)