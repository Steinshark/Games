from __future__ import print_function
#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset,DataLoader 
from torch.nn.functional import interpolate
import math 
from torchvision.utils import make_grid


class preload_ds(Dataset):

    def __init__(self,fname_l,processor):
        self.data       = fname_l
        self.processor  = processor

    #Return tensor, fname
    def __getitem__(self,i):
        if not self.processor is None:
            tensor  = self.processor(torch.load(self.data[i]))
        else:
            tensor  = torch.load(self.data[i])
        return tensor
    def __len__(self):
        return len(self.data)

def load_locals(bs=8,processor=lambda x: x,local_dataset_path="C:/data/images/converted_tensors/"):
    dataset     = preload_ds([local_dataset_path+f for f in os.listdir(local_dataset_path)],processor=processor)
    return DataLoader(dataset,batch_size=bs,shuffle=True)

def downsample(img:torch.Tensor,stage=0):
    if stage == 0:
        return interpolate(img,size=(32,48))
    elif stage == 1:
        return interpolate(img,size=(64,96))
    elif stage == 2:
        return interpolate(img,size=(128,192))
    elif stage == 3:
        return interpolate(img,size=(256,384))
    elif stage >= 4:
        return img

def fix_img(img:torch.Tensor,mode="tanh"):
    if mode == "tanh":
        img += 1 
        img /= 2 

    return img  

from models import *
#from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 69
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)



# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 256

# Size of feature maps in generator
n_f     = 64

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 150

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# N times sampled for images 
sample_every    = 100

# N times updated on cmd line 
update_every    = 1000       

# N imgs sampled 
n_samples       = 36 

# Train stage   
train_stage     = 0




# We can use an image folder dataset the way we have it setup.
# Create the dataset
# Create the dataloader
dataloader = load_locals(batch_size)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




# Create the generator
netG = generator(nz,n_f).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
#print(netG)

# Create the Discriminator
netD = discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Load models 
#d_state_dict    = torch.load(f"C:/gitrepos/projects/ml/image/models/netD{16}.model")
#g_state_dict    = torch.load(f"C:/gitrepos/projects/ml/image/models/netG{16}.model")
#netD.load_state_dict(d_state_dict)
#netG.load_state_dict(g_state_dict)
# Print the model
#print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(1, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.AdamW(netD.model.parameters(), lr=lr/10, betas=(beta1, 0.999))
optimizerG = optim.AdamW(netG.model.parameters(), lr=lr, betas=(beta1, 0.999))


# Set stage 
stages          = {
            0 : {"netD_fwd":netD.forward1,
                 "netG_fwd":netG.forward1},
            1 : {"netD_fwd":netD.forward2,
                 "netG_fwd":netG.forward2},
            2 : {"netD_fwd":netD.forward3,
                 "netG_fwd":netG.forward3}
}

d_fw_fn     = stages[train_stage]["netD_fwd"]
g_fw_fn     = stages[train_stage]["netG_fwd"]

inp         = torch.randn(size=(1,nz,1,1),device=device)
print(f"Random Seed:\t{manualSeed}")
print(f"latent_space:\t{nz}")
print(f"batch size:\t{batch_size}")
g_out       = g_fw_fn(inp)
print(f"g out size:\t{tuple(g_out.shape)}")
print(f"d out size:\t{tuple(d_fw_fn(g_out).shape)}")
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print(f"\nStarting Training Loop stage {train_stage}...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    real_imgs   = [] 
    fake_imgs   = [] 

    for i, data in enumerate(dataloader, 0):
        
        #Convert To Proper scaling 
        real_cpu = downsample(data.to(device).float(),stage=train_stage)
            
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        
        for param in netD.model.parameters():
            param.grad  = None 

        # Format batch
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = d_fw_fn(real_cpu).view(-1)

        #input(f"output is size {output.shape}")
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = g_fw_fn(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = d_fw_fn(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        if errD.item() > .5:
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        for param in netG.model.parameters():
            param.grad  = None 
        #netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = d_fw_fn(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        if errG.item() > .5:
            optimizerG.step()


        if len(fake_imgs) < n_samples:
            real_imgs.append(fix_img(real_cpu[0].detach().cpu()))
            fake_imgs.append(fix_img(fake[0].detach().cpu()))

        # Output training stats
        if i % update_every == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.3f\tLoss_G: %.3f\tD(x): %.3f\tD(G(z)): %.3f / %.3f\t%d/%d images'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,batch_size*i,batch_size*len(dataloader)))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())


        if i % sample_every == 0 and i > 0:

            paired      = list(zip(real_imgs,fake_imgs)) 
            with torch.no_grad():
                img_grid    = [] + real_imgs + [img for img in g_fw_fn(torch.randn(size=(n_samples,nz,1,1),device=device)).detach().cpu()] 

            slice       = int(math.sqrt(n_samples))

            grid        = make_grid(img_grid,nrow=slice)
            display     = transforms.ToPILImage()(grid)
            #display.show()
            display.save(f"C:/gitrepos/projects/ml/image/gan/imgs{epoch}_{i}.jpg")
            img_grid = [] 
            real_imgs   = [] 
            fake_imgs   = [] 

        iters += 1  
    
    print('[%d/%d][%d/%d]\tLoss_D: %.3f\tLoss_G: %.3f\tD(x): %.3f\tD(G(z)): %.3f / %.3f\t%d/%d images'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,batch_size*i,batch_size*len(dataloader)))
    
    paired      = list(zip(real_imgs,fake_imgs)) 
    with torch.no_grad():
        img_grid    = [] + real_imgs + [img for img in g_fw_fn(torch.randn(size=(n_samples,nz,1,1),device=device)).detach().cpu()] 

    slice       = int(math.sqrt(n_samples))

    grid        = make_grid(img_grid,nrow=slice)
    display     = transforms.ToPILImage()(grid)
    #display.show()
    display.save(f"C:/gitrepos/projects/ml/image/gan/imgs{epoch}_{i}.jpg")
    img_grid = [] 
    real_imgs   = [] 
    fake_imgs   = []

    # Save models 
    torch.save(netD.state_dict(),f"C:/gitrepos/projects/ml/image/models/netD{epoch}.model")
    torch.save(netG.state_dict(),f"C:/gitrepos/projects/ml/image/models/netG{epoch}.model")