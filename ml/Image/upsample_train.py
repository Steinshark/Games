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


def tfloat_to_np(x:torch.Tensor)->numpy.ndarray:
    
    #convert to 1-255
    x           = x + 1 
    x           = x / 2
    x           = x * 255
    x           = x.int()
    x           = x.numpy()
    x           = numpy.transpose(x,[1,2,0])
    return x 


def make_grid(x:torch.Tensor,y:torch.Tensor,ep:int,batch:int)->None:

    #Take first 3 imgs 
    width       = 2* y.shape[-2]
    height      = 3* y.shape[-1]

    #Refurbish1 
    x1              = x[0].unsqueeze_(dim=0)
    x1              = torch.nn.functional.interpolate(x1,scale_factor=4)[0]
    refurbished1    = torch.ones(size=(y.shape[1:]))
    refurbished1[:,:x1.shape[1],:x1.shape[2]]    = x1
    row1            = torch.cat([refurbished1,y[0]],dim=1)

    #Refurbish2
    x2              = x[1].unsqueeze_(dim=0)
    x2              = torch.nn.functional.interpolate(x2,scale_factor=4)[0]
    refurbished2    = torch.ones(size=(y.shape[1:]))
    refurbished2[:,:x2.shape[1],:x2.shape[2]]    = x2
    row2            = torch.cat([refurbished2,y[1]],dim=1)

    #Refurbish3
    x3              = x[2].unsqueeze_(dim=0)
    x3              = torch.nn.functional.interpolate(x3,scale_factor=4)[0]
    refurbished3    = torch.ones(size=(y.shape[1:]))
    refurbished3[:,:x3.shape[1],:x3.shape[2]]    = x3
    row3            = torch.cat([refurbished3,y[2]],dim=1)

    finalimg        = torch.cat([row1,row2,row3],dim=2)



    plt.imshow(tfloat_to_np(finalimg))
    plt.savefig(f"C:/code/projects/ml/Image/outs/ep{ep}_b{batch}")

    
def train(ds_path,n_ep,bs):

    #LOCAL OPTIONS
    start_dim           = 48 
    max_distorts        = 4
    max_strength        = .1
    max_n               = 256

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
    dataloader      =   load_locals(bs=bs,processor=None,max_n=max_n)
    
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
            x,y         = apply_distortion(base_img,noise_sf=random.random()*max_strength,noise_iters=random.randint(0,max_distorts))    
            #Send thorugh model 
            pred        =  model.forward(x)

            #Loss
            loss        = loss_fn(pred,y)
            losses[-1].append(loss.mean().item())
            loss.backward()

            #Grad 
            optim.step()
            make_grid(x,pred,ep,batch_i)
            print(f"\tbatch loss={losses[-1][-1]:.5f}")
            

if __name__ == '__main__':

    train("C:/data/images/converted_tensors/",4,8) 
        