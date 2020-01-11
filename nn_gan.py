import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import os
# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 8

nearn_gen_size = 32

nearn_comp_size = 48

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self,input=None):
        if input is None:
            input = torch.randn(batch_size, nz, 1, 1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return torch.squeeze(self.main(input))

def batch_euclid_distance(A,B):

    A2 = torch.einsum('ai, ai -> a', A, A)
    B2 = torch.einsum('bi, bi -> b', B, B)
    AB = torch.einsum('ai, bi -> ab', A, B)
    A2 = torch.unsqueeze(A2,dim=1)
    return (A2 - 2.0*AB) + B2

class Trainer:
    def __init__(self,data_path):
        self.data_path = data_path
        self.gen = Generator()
        self.discrim = Discriminator()

        self.discrim_optimizer = torch.optim.Adam(self.discrim.parameters(), lr=lr, betas=(beta1, 0.999))
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))

    def gen_train_step(self):
        tot_loss = 0
        for x in range(nearn_gen_size//batch_size):
            gen_imgs = self.gen_train_data()
            discrim_logits = self.discrim(gen_imgs)
            loss = -torch.mean(discrim_logits)
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()
            tot_loss += loss.cpu().detach()
        return tot_loss

    def discrim_train_step(self,data):
        with torch.no_grad():
            gen_imgs = self.gen_train_data()

            true_imgs = data
            true_labels = self.calc_nearest_neighbor_scores(true_imgs,gen_imgs)

        tot_loss = 0
        for x in range(nearn_gen_size//batch_size):
            self.discrim_optimizer.zero_grad()
            batch_predicted_logits = self.discrim(gen_imgs[x*batch_size:(x+1)*batch_size])
            batch_true_vals = true_labels[x*batch_size:(x+1)*batch_size]

            loss = torch.nn.BCEWithLogitsLoss()(batch_predicted_logits,batch_true_vals)
            loss.backward()
            self.discrim_optimizer.step()
            tot_loss += loss.cpu().detach()

        return tot_loss / (nearn_gen_size / batch_size)

    def gen_train_data(self):
        imgs = []
        for b in range(0,nearn_gen_size,batch_size):
            imgs.append(self.gen())
        return torch.cat(imgs,dim=0)

    def calc_nearest_neighbor_scores(self,true_imgs,gen_imgs):
        flat_gens = torch.flatten(gen_imgs,start_dim=1)
        flat_trues = torch.flatten(true_imgs,start_dim=1)

        distances = batch_euclid_distance(flat_trues,flat_gens)
        maxarg = torch.argmax(distances,dim=1)
        img_scores = torch.zeros(nearn_gen_size)
        img_scores[maxarg] = 1.0
        return img_scores

    def train(self, data_loader):
        tot_gen_loss = 0
        tot_discrim_loss = 0
        BATCHES_PER_PRINT = 10
        for batch_idx, (data, target) in enumerate(data_loader):
            tot_discrim_loss += self.discrim_train_step(data)
            tot_gen_loss += self.gen_train_step()
            if batch_idx % BATCHES_PER_PRINT == 0:
                print(tot_discrim_loss / BATCHES_PER_PRINT,"\t\t",tot_gen_loss / BATCHES_PER_PRINT)
                tot_gen_loss = 0
                tot_discrim_loss = 0

    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder(
            root=self.data_path,
            transform=torchvision.transforms.ToTensor()
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=nearn_comp_size,
            num_workers=0,
            shuffle=True
        )
        return train_loader

def main():
    trainer = Trainer(os.path.abspath("data/train_imgs/"))
    loader = trainer.train_dataloader()
    trainer.train(loader)

if __name__ == "__main__":
    main()
