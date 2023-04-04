#Author Everett Stenberg - github: Steinshark

#Description 
# Create a GAN for image generation of 128x128
# Currently used for images of bread in my kitchen 

import torch 
from torch.nn import Linear, ConvTranspose2d, BatchNorm2d, LeakyReLU, Tanh, MaxPool2d, Upsample, Conv2d, Sigmoid, ReLU, Dropout2d
from torch.optim import Adam 
from torch.nn import BCELoss
import os 
from torch import nn 
import random 
import time 
from torchvision.io import read_image
from torchvision.io import read_video
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize,CenterCrop,ToTensor,Normalize
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt 
import numpy 
from PIL import Image

############################################################################################################################
############################################          HELPER FUNCTIONS          ############################################ 
############################################################################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def picturize_vid(filename):
    save_path           = "C:/data/bread/bread_pics"
    startnum            = max([int(f.replace(".jpg","")) for f in os.listdir(save_path)]) + 1
    vid                 = read_video(filename,pts_unit='sec')[0].type(torch.uint8)
    for i in range(vid.shape[0]): 
        im_arr              = vid[i].numpy()
        img                 = Image.fromarray(im_arr)
        img.save(f"C:/data/bread/bread_pics/{startnum+i}.jpg")

def save_model(model:torch.nn.Module,savepath:str):
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(),savepath)

def load_model(model:torch.nn.Module,savepath):
    try:
        model.load_state_dict(torch.load(savepath))
    except FileNotFoundError:
        print(f"{savepath} does not exist!")

def sample(g_model,lvs,n_imgs,path="samples/IMGS_0",title="bread"):
    with torch.no_grad():
        ex                  = vutils.make_grid(g_model.forward(torch.randn(size=(n_imgs,lvs,1,1),dtype=torch.float,device=dev)).detach().cpu(),padding=1,normalize=True)

        # Plot the fake images from the last epoch
        fig,axs     = plt.subplots(nrows=1,ncols=1)
        axs.axis    = 'off'
        fig.suptitle(title)
        axs.imshow(numpy.transpose(ex,(1,2,0)))
        img         = plt.gcf()
        img.set_size_inches(30,16)
        img.savefig(f"{path}",dpi=100)
        plt.cla()
        plt.close()

def update_lr(optimizer:torch.optim.Optimizer,sf,thresh):
    old_lr                          = optimizer.param_groups[0]['lr']
    new_lr                          = old_lr * sf 
    optimizer.param_groups[0]['lr'] = max(new_lr,thresh)



############################################################################################################################
############################################           USEFUL SETTINGS          ############################################ 
############################################################################################################################

#Training parameters 
bs                      = 32
im_len                  = 128
latent_vector_size      = 512
neg_slope               = .02
dev                     = torch.device('cuda')
epochs                  = 100 
root                    = "C:/data/bread"
im_dset                 = ImageFolder(root=root,transform=Compose([Resize(128),CenterCrop(128),ToTensor(),Normalize((.5,.5,.5),(.5,.5,.5))]))
print(f"Dataset size: {im_dset.__len__()}")
use_bias                = False
droprate                = .1
ch_layers               = [512,512,256,256,256,256,128,64]
GMODEL_SAVEPATH          = "models/Gmodel_state_dict"           # Used for saving model 
DMODEL_SAVEPATH          = "models/Dmodel_state_dict"
torch.backends.cudnn.benchmark = True

#Optimizer settings
g_lr                    = .00002                # lr for generator 
g_lr_thresh             = .00001                # Used to set lower bound for lr scheduler
g_betas                 = (.5,.999)
g_wd                    = .00001                # Weight decay of generator optimizer
dlg_dt                   = .9                   # Scale factor for lr scheduler

d_lr                    = .00002
d_lr_thresh             = .00001
d_betas                 = (.5,.999)
d_wd                    = .00001
dld_dt                   = .9

############################################################################################################################
############################################         Model Architecture         ############################################ 
############################################################################################################################

class GModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            
            #Upsample x2                (1x1) -> (2x2)
            Upsample(scale_factor=2),
            Conv2d(in_channels=latent_vector_size,out_channels=ch_layers[0],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[0]),
            ReLU(inplace=True),

            #Upsample x2                (2x2) -> (4x4)
            Upsample(scale_factor=2),
            Conv2d(in_channels=ch_layers[0],out_channels=ch_layers[1],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[1]),
            ReLU(inplace=True),

            #Upsample x2                (4x4) -> (8x8)
            Upsample(scale_factor=2),
            Conv2d(in_channels=ch_layers[1],out_channels=ch_layers[2],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[2]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[2],out_channels=ch_layers[2],kernel_size=5,stride=1,padding=2,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[2]),
            ReLU(inplace=True),

            #Upsample x2                (8x8) -> (16x16)
            Upsample(scale_factor=2),
            Conv2d(in_channels=ch_layers[2],out_channels=ch_layers[3],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[3]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[3],out_channels=ch_layers[3],kernel_size=5,stride=1,padding=2,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[3]),
            ReLU(inplace=True),
   
            #Upsample x2                (16x16) -> (32x32)
            Upsample(scale_factor=2),
            Conv2d(in_channels=ch_layers[3],out_channels=ch_layers[4],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[4]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[4],out_channels=ch_layers[4],kernel_size=5,stride=1,padding=2,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[4]),
            ReLU(inplace=True),

            #Upsample x2                (32x32) -> (64x64)
            Upsample(scale_factor=2),
            Conv2d(in_channels=ch_layers[4],out_channels=ch_layers[6],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[6]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[6],out_channels=ch_layers[6],kernel_size=5,stride=1,padding=2,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[6]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[6],out_channels=ch_layers[6],kernel_size=5,stride=1,padding=2,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[6]),
            ReLU(inplace=True),

            #Upsample x2                (64x64) -> (128x128)
            Upsample(scale_factor=2),
            Conv2d(in_channels=ch_layers[6],out_channels=ch_layers[7],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[7]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[7],out_channels=ch_layers[7],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[7]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[7],out_channels=ch_layers[7],kernel_size=3,stride=1,padding=1,bias=use_bias),
            Dropout2d(p=droprate),
            BatchNorm2d(ch_layers[7]),
            ReLU(inplace=True),
            Conv2d(in_channels=ch_layers[7],out_channels=3,kernel_size=3,stride=1,padding=1,bias=use_bias),
            #Activate 
            #LeakyReLU(negative_slope=neg_slope,inplace=True),
            #BatchNorm 
            #BatchNorm2d(128),

            # #Upsample x2 
            # Upsample(scale_factor=2),
            # #Conv 
            # Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            # #Activate 
            # LeakyReLU(negative_slope=neg_slope,inplace=True),
            # #BatchNorm 
            # BatchNorm2d(128),
            # #Conv 
            # Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1),
            # #Activate 
            # LeakyReLU(negative_slope=neg_slope,inplace=True),
            # #BatchNorm 
            # BatchNorm2d(64),
            # #Conv 
            # Conv2d(in_channels=64,out_channels=3,kernel_size=5,stride=1,padding=2),
            # #Activate 
            Tanh()
        )
        self.model = self.model.to(dev)
    
    def forward(self,x) -> torch.tensor:
        return self.model(x)


class GModel2(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            ConvTranspose2d(latent_vector_size,1024,4,1,0,bias=False),
            BatchNorm2d(1024),
            ReLU(True),
            #4x4
            ConvTranspose2d(1024,512,4,2,1,bias=False),
            ReLU(True),
            Conv2d(512,512,3,1,1,bias=use_bias),
            BatchNorm2d(512),
            ReLU(True),
            #16x16
            ConvTranspose2d(512,512,4,2,1,bias=False),
            BatchNorm2d(512),
            ReLU(True),
            #32x32
            ConvTranspose2d(512,512,4,2,1,bias=False),
            BatchNorm2d(512),
            ReLU(True),
            Conv2d(512,256,5,1,2,bias=use_bias),
            BatchNorm2d(256),
            ReLU(True),
            #64x64
            ConvTranspose2d(256,256,4,2,1,bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256,256,3,1,1,bias=use_bias),
            BatchNorm2d(256),
            ReLU(True),
            Conv2d(256,128,5,1,2,bias=use_bias),
            BatchNorm2d(128),
            ReLU(True),
            #128x128
            ConvTranspose2d(128,64,4,2,1,bias=False),
            BatchNorm2d(64),
            ReLU(True),
            Conv2d(64,32,5,1,2,bias=use_bias),
            BatchNorm2d(32),
            ReLU(True),
            Conv2d(32,32,5,1,2,bias=False),
            BatchNorm2d(32),
            ReLU(True),
            Conv2d(32,3,3,1,1,bias=False),
            Tanh()
        )
        self.model = self.model.to(dev)
    
    def forward(self,x) -> torch.tensor:
        return self.model(x)


class DModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            #INPUT (3,256,256)
            Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(64),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(128),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            Conv2d(in_channels=128,out_channels=256,kernel_size=5,stride=1,padding=2,bias=use_bias),
            BatchNorm2d(256),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            MaxPool2d(kernel_size=2,stride=2),

            #INPUT (64,128,128)
            Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(256),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(512),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            Conv2d(in_channels=512,out_channels=512,kernel_size=5,stride=1,padding=2,bias=use_bias),
            BatchNorm2d(512),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            MaxPool2d(kernel_size=2,stride=2),

            #INPUT (256,64,64)
            Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(512),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            Conv2d(in_channels=512,out_channels=512,kernel_size=5,stride=1,padding=2,bias=use_bias),
            BatchNorm2d(512),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            MaxPool2d(kernel_size=2,stride=2),

            #INPUT (256,32,32)
            Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(512),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            Conv2d(in_channels=512,out_channels=512,kernel_size=5,stride=1,padding=2,bias=use_bias),
            BatchNorm2d(512),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            MaxPool2d(kernel_size=2,stride=2),

            #INPUT (256,16,16)
            Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(1024),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(1024),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            MaxPool2d(kernel_size=2,stride=2),

            #INPUT (512,8,8)
            Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1,bias=use_bias),
            BatchNorm2d(1024),
            LeakyReLU(negative_slope=neg_slope,inplace=True),
            MaxPool2d(kernel_size=2,stride=2),

            #INPUT (512,4,4)
            Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1,bias=use_bias),

            #INPUT (512,2,2)
            # Conv2d(in_channels=512,out_channels=1,kernel_size=3,stride=1,padding=1),
            # LeakyReLU(negative_slope=neg_slope,inplace=True),
            # MaxPool2d(kernel_size=2,stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(4096,512),
            torch.nn.Dropout(p=.1),
            torch.nn.LeakyReLU(negative_slope=neg_slope),
            torch.nn.Linear(512,64),
            torch.nn.Dropout(p=.1),
            torch.nn.LeakyReLU(negative_slope=neg_slope),
            torch.nn.Linear(64,1),
            Sigmoid()
        )

        self.model.to(dev)

    def forward(self,x):
        return self.model(x)

############################################################################################################################
############################################           Training Init            ############################################ 
############################################################################################################################

Generator               = GModel()
Discriminator           = DModel()
Generator.apply(weights_init)
Discriminator.apply(weights_init)

#load_model(Generator,GMODEL_SAVEPATH)
#load_model(Discriminator,DMODEL_SAVEPATH)

g_optim                 = Adam(Generator.parameters(),      lr=g_lr,betas=g_betas,weight_decay=g_wd)
d_optim                 = Adam(Discriminator.parameters(),  lr=d_lr,betas=d_betas,weight_decay=d_wd)

loss_fn                 = BCELoss()

print(f"Created models\tG-{(sum([p.nelement() * p.element_size() for p in Generator.parameters()])/1000000):.2f}MB\tD-{(sum([p.nelement() * p.element_size() for p in Discriminator.parameters()])/1000000):.2f}MB")
in_vect                 = torch.ones(size=(2,latent_vector_size,1,1),dtype=torch.float32,device=dev)
print(f"passing {in_vect.shape} -> {Generator.forward(in_vect).shape}")
out_vect                = Generator.forward(in_vect)
print(f"passing {out_vect.shape} -> {Discriminator.forward(out_vect).shape}")


############################################################################################################################
############################################            Dataset Init            ############################################ 
############################################################################################################################

class IDataset(Dataset):

    def __init__(self,imgs,limit=100):
        self.data   = []

        for i,file in enumerate(imgs):
            if i > limit:
                break 
            img         = read_image(os.path.join(root,file)).type(torch.uint8)
            if not img.shape == (3,256,256):
                pass
            else:
                self.data.append(img)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i],1

class VDataset(Dataset):

    def __init__(self,videoTensor):
        self.data   = videoTensor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,i):
        return self.data[i],1


dataloader              = DataLoader(im_dset,batch_size=bs,shuffle=True)


for ep in range(epochs):

    losses              = [] 
    t0                  = time.time()
    

    for i,data in enumerate(dataloader):
        tb                  = time.time()
        #Clear gradients    
        for d_p in Discriminator.parameters():
            d_p.grad        = None 

        imgs_real           = data[0].to(dev).type(torch.float32)
        real_labels         = torch.ones(size=(len(data[1]),1),device=dev)
        gen_labels          = torch.zeros(size=(len(data[1]),1),device=dev)

        #Train Discriminator on real
        predictions_real    = Discriminator.forward(imgs_real)
        loss_real           = loss_fn(predictions_real,real_labels)
        losses.append([loss_real.item(),0,0])
        loss_real.backward() 

        #Train Discriminator on fake 
        imgs_fake           = Generator.forward(torch.randn(size=(len(data[1]),latent_vector_size,1,1),dtype=torch.float,device=dev))
        predictions_fake    = Discriminator.forward(imgs_fake.detach())
        loss_fake           = loss_fn(predictions_fake,gen_labels)
        losses[-1][1]       = loss_fake.item()
        loss_fake.backward() 

        #Update weights 
        d_optim.step()

        #Train Generator 
        for g_p in Generator.parameters():
            g_p.grad            = None 

        predictions_fake_t  = Discriminator.forward(imgs_fake)
        loss_fake_t         = loss_fn(predictions_fake_t,real_labels)
        losses[-1][-1]      = loss_fake_t.item()
        loss_fake_t.backward()
        g_optim.step()

        if i % 500 == 0:
            sample(Generator,latent_vector_size,64,path=f"samples/ep{ep}_batch{i}",title=f"Bread imgs ep{ep} batch{i}")
            print(f"\tbatch {i}/{dataloader.__len__()}\tep_t:{(time.time()-t0):.2f}s\tbatch_t:{(time.time()-tb):.2f}s")
            update_lr(g_optim,dlg_dt,g_lr_thresh)
            update_lr(d_optim,dld_dt,d_lr_thresh)

    
    #Save model states 
    save_model(Generator,GMODEL_SAVEPATH)
    save_model(Discriminator,DMODEL_SAVEPATH)

    #Update lrs 
    update_lr(g_optim,dlg_dt,g_lr_thresh)
    update_lr(d_optim,dld_dt,d_lr_thresh)
    print(f"Epoch\t{ep} - lossD_real:{(sum(l[0] for l in losses)/i):.3f} - lossD_fake:{(sum(l[1] for l in losses)/i):.3f} - lossG_fake:{(sum(l[2] for l in losses)/i):.3f} - t:{(time.time()-t0):.2f}s\t - lr:{g_optim.param_groups[0]['lr']:.5f}")
        




