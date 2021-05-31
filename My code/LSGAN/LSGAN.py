import torch
import torch.nn as nn

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.fc =nn.Sequential(
            nn.Linear(latent_dim, 4*4*128),
            nn.BatchNorm2d(4*4*128, 0.8),
            nn.ReLU(),
        )
        self.gen = nn.Sequential(
            # ConvTranspose2d(입력채널수, 결과채널수, fitler크기, stride, padding)
            # out_dim = stride*(input_dim−1)+filter−2∗padding
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=0),
            # |out| = (batch, channel, 10, 10)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
            # |out| = (batch, channel, 13, 13)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=0),
            # |out| = (batch, channel, 28, 28)
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 128, 4, 4)
        img = self.gen(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 28 // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity       