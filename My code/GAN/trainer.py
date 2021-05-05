import torch
import torch.optim as optim

class Trainer:
    def __init__(self,
            generator, discriminator,
            crit,
            config):
        self.generator = generator.to(config.device)
        self.discriminator = discriminator.to(config.device)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.001)
        self.crit = crit
        self.config = config

    def train(self, train_loader, valid_loader):
        for epoch in range(self.config.epochs):
            self.generator.train()
            self.discriminator.train()
            
            for imgs, _ in range(train_loader):
                imgs = imgs.reshape(self.config.batch_size, -1).to(self.config.device)
                real_label = torch.ones(self.config.batch_size, 1).to(self.config.device)
                fake_label = torch.zeros(self.config.batch_size, 1).to(self.config.device)
                # train discriminator
                self.optimizer_D.zero_grad()
                z = torch.normal(mean=0, std=1, size=(imgs.shape[0], self.config.latent_dim)).to(self.config.device)
                fake_imgs = self.generator(z)
                outputs = self.discriminator(fake_imgs)
                d_loss_fake = self.crit(outputs, fake_label)

                outputs = self.discriminator(imgs)
                d_loss_real = self.crit(outputs, real_label)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_D.step()

                # train generator
                self.optimizer_G.zero_grad()
                z = torch.normal(mean=0, std=1, size=(imgs.shape[0], self.config.latent_dim)).to(self.config.device)
                fake_imgs = self.generator(z)
                outputs = self.discriminator(fake_imgs)
                g_loss = self.crit(outputs, real_label)
                g_loss.backward()
                self.optimizer_G.step()

            # valid
            self.generator.eval()
            self.discriminator.eval()
            for imgs, _ in range(valid_loader):
                with torch.no_grad():
                    real_label = torch.ones(self.config.batch_size, 1).to(self.config.device)
                    fake_label = torch.zeros(self.config.batch_size, 1).to(self.config.device)
                    imgs = imgs.reshape(self.config.batch_size, -1).to(self.config.device)
                    
                    z = torch.normal(mean=0, std=1, size=(imgs.shape[0], self.config.latent_dim)).to(self.config.device)
                    fake_images = self.generator(z)
                    outputs = self.discriminator(fake_images)
                    d_loss_fake = self.crit(outputs, fake_label)
                    outputs = self.discriminator(imgs)
                    d_loss_real = self.crit(outputs, real_label)
                    d_loss_val = d_loss_real + d_loss_fake

                    z = torch.normal(mean=0, std=1, size=(imgs.shape[0], self.config.latent_dim)).to(self.config.device)
                    fake_images = self.generator(z)
                    outputs = self.discriminator(fake_images)
                    g_loss_val = self.crit(outputs, real_label)

            # 하나의 epoch이 끝날 때마다 로그(log) 출력
            print(f"[Epoch {epoch}/{self.config.epochs}] [D train loss: {d_loss.item():.6f}] [G train loss: {g_loss.item():.6f}]")
            print(f"[Epoch {epoch}/{self.config.epochs}] [D val loss: {d_loss_val.item():.6f}] [G val loss: {g_loss_val.item():.6f}]")
        
        # model save
        torch.save(self.generator.state_dict(), 'G.pt')
        torch.save(self.generator.state_dict(), 'D.pt')