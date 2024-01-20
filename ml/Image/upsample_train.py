import torch 
from torch.utils.data import Dataset,DataLoader
import numpy 
from matplotlib import pyplot as plt 
import random 
import time 
from gen import load_locals,preload_ds


DEV                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UpSampleNet(torch.nn.Module):


    def __init__(self,start_dim:int,final_dim:int):

        #Super init
        super(UpSampleNet,self).__init__()

        #Define vars 
        conv_act            = torch.nn.GELU
        up_act              = torch.nn.LeakyReLU
        final_act           = torch.nn.LeakyReLU

        #Create modules
        self.convLayers     = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,3,1,1),
            conv_act(),
            torch.nn.Conv2d(8,32,3,1,1),
            conv_act(),
        )

        #Make 2 jumps up to final size - linearly 

        first_dim           = int(start_dim + (final_dim-start_dim)/2) 
        self.upLayers       = torch.nn.Sequential(
            torch.nn.Upsample(size=(first_dim,first_dim)),
            torch.nn.Conv2d(32,16,5,1,2),
            up_act(),

            torch.nn.Upsample(size=(final_dim,final_dim)),
            torch.nn.Conv2d(16,8,5,1,2),
            up_act(),
        )

        self.finalLayers    = torch.nn.Sequential(
            torch.nn.Conv2d(8,4,9,1,4),
            final_act(),

            torch.nn.Conv2d(4,3,17,1,8),
            torch.nn.Tanh()
        )


    def forward(self,x:torch.Tensor)->torch.Tensor():
        x           = self.convLayers(x)
        x           = self.upLayers(x)
        x           = self.finalLayers(x)
        return x
    

def apply_distortion(img:torch.Tensor,start_dim:int=48,noise_sf:float=.1,noise_iters:int=4) -> torch.Tensor:

    #Downscale 
    sf              = start_dim / img.shape[-1]
    downscaled      = torch.nn.functional.interpolate(img,scale_factor=sf)

    #Apply noise
    for _ in range(noise_iters):
        downscaled  = downscaled + torch.randn(size=downscaled.shape,device=downscaled.device) * noise_sf

    downscaled      = torch.clip(downscaled,-1,1)
    # plt.imshow(numpy.transpose(((img[0]+1)/2).numpy(),[1,2,0]))
    # plt.show()
    # plt.imshow(numpy.transpose(((downscaled[0]+1)/2).numpy(),[1,2,0]))
    # plt.show()

    return downscaled,img


def train(ds_path,n_ep,bs):

    #LOCAL OPTIONS
    start_dim           = 48 
    max_distorts        = 4
    max_strength        = .1

    #TRAIN OPTIONS 
    lr                  = 1e-4
    wd                  = .01
    betas               = (.5,.9993)
    bs                  = 8


    #Create model and optim
    model               = UpSampleNet(start_dim=start_dim,final_dim=512).to(DEV)
    optim               = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd,betas=betas)
    loss_fn             = torch.nn.MSELoss()

    #Prepare data
    dataloader      =   load_locals(bs=bs,processor=None,max_n=128)
    
    #Stats
    losses              = []

    for ep in range(n_ep):
        
        print(f"\n\tEPOCH {ep}")
        losses.append([])    

        #Run train loop
        for batch_i,item in enumerate(dataloader):

            #Zero
            optim.zero_grad()

            #Get base img
            base_img    = item.to(DEV).type(torch.float)

            #Apply distortion
            x,y         = apply_distortion(base_img)    

            #Send thorugh model 
            pred        =  model.forward(x)

            #Loss
            loss        = loss_fn(pred,y)
            losses[-1].append(loss.mean().item())
            loss.backward()

            #Grad 
            optim.step()

            print(f"\tbatch loss={losses[-1][-1]:.5f}")

if __name__ == '__main__':

    train("C:/data/images/converted_tensors/",4,8) 
        