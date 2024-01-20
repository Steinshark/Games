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
            torch.nn.Conv2d(3,16,3,1,1),
            conv_act(),
            torch.nn.Conv2d(16,48,3,1,1),
            conv_act(),
        )

        #Make 2 jumps up to final size - linearly 

        first_dim           = int(start_dim + (final_dim-start_dim)/2) 
        self.upLayers       = torch.nn.Sequential(
            torch.nn.Upsample(size=(first_dim,first_dim)),
            torch.nn.Conv2d(48,32,5,1,2),
            up_act(),

            torch.nn.Upsample(size=(final_dim,final_dim)),
            torch.nn.Conv2d(32,12,5,1,2),
            up_act(),
        )

        self.finalLayers    = torch.nn.Sequential(
            torch.nn.Conv2d(12,8,7,1,3),
            final_act(),

            torch.nn.Conv2d(8,3,9,1,4),
            torch.nn.Tanh()
        )


    def forward(self,x:torch.Tensor)->torch.Tensor():
        x           = self.convLayers(x)
        x           = self.upLayers(x)
        x           = self.finalLayers(x)
        return x
    

class DiscrNet(torch.nn.Module):

    def __init__(self,in_dim:int):

        #Super init
        super(DiscrNet,self).__init__()

        #Define vars 
        conv_act            = torch.nn.LeakyReLU
        lin_act             = torch.nn.LeakyReLU
        final_act           = torch.nn.LeakyReLU

        linear_size         = int(in_dim / 32)

        #Create modules
        self.convLayers     = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,1,1),
            conv_act(),

            torch.nn.Conv2d(16,32,5,2,2),               #   /2  
            conv_act(),

            torch.nn.Conv2d(32,64,5,2,2),               #   /4
            conv_act(),

            torch.nn.Conv2d(64,128,5,2,2),              #   /8 
            conv_act(),

            torch.nn.Conv2d(128,128,5,2,2),             #   /16
            conv_act(),

            torch.nn.Conv2d(128,128,5,2,2),             #   /32 128:4   256:8  512:16
            conv_act(),
            torch.nn.Flatten(1),
        )

        self.linearLayers   = torch.nn.Sequential(
            torch.nn.Linear(128 * linear_size * linear_size,256),
            torch.nn.Dropout(p=.2),
            lin_act(),

            torch.nn.Linear(256,64),
            torch.nn.Dropout(p=.1),
            lin_act(),

            torch.nn.Linear(64,1),
            torch.nn.Sigmoid()
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x           = self.convLayers(x)
        x           = self.linearLayers(x)
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

    return downscaled


def tfloat_to_np(x:torch.Tensor)->numpy.ndarray:
    
    #convert to 1-255
    x           = x + 1 
    x           = x / 2
    x           = x * 255
    x           = x.int()
    x           = x.numpy()
    x           = numpy.transpose(x,[1,2,0])
    return x 


def make_grid(x:torch.Tensor,y:torch.Tensor,ep:int,batch:int,sf=1)->None:

    #Take first 3 imgs 
    width       = 2* y.shape[-2]
    height      = 3* y.shape[-1]

    #Refurbish1 
    x1              = x[0].unsqueeze_(dim=0)
    x1              = torch.nn.functional.interpolate(x1,scale_factor=sf)[0]
    refurbished1    = torch.ones(size=(y.shape[1:]))
    refurbished1[:,:x1.shape[1],:x1.shape[2]]    = x1
    row1            = torch.cat([refurbished1,y[0]],dim=1)

    #Refurbish2
    x2              = x[1].unsqueeze_(dim=0)
    x2              = torch.nn.functional.interpolate(x2,scale_factor=sf)[0]
    refurbished2    = torch.ones(size=(y.shape[1:]))
    refurbished2[:,:x2.shape[1],:x2.shape[2]]    = x2
    row2            = torch.cat([refurbished2,y[1]],dim=1)

    #Refurbish3
    x3              = x[2].unsqueeze_(dim=0)
    x3              = torch.nn.functional.interpolate(x3,scale_factor=sf)[0]
    refurbished3    = torch.ones(size=(y.shape[1:]))
    refurbished3[:,:x3.shape[1],:x3.shape[2]]    = x3
    row3            = torch.cat([refurbished3,y[2]],dim=1)

    finalimg        = torch.cat([row1,row2,row3],dim=2)



    plt.imshow(tfloat_to_np(finalimg))
    plt.savefig(f"C:/code/projects/ml/Image/outs/ep{ep}_b{batch}")

    
def train(ds_path,n_ep,bs):

    #LOCAL OPTIONS
    start_dim           = 48 
    final_dim           = 128
    max_distorts        = 4
    max_strength        = .1
    max_n               = 256

    #TRAIN OPTIONS 
    lr                  = 1e-4
    wd                  = .001
    betas               = (.75,.9993)
    bs                  = bs


    #Create model and optim
    model               = UpSampleNet(start_dim=start_dim,final_dim=128).to(DEV)
    optim               = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd,betas=betas)
    loss_fn             = torch.nn.MSELoss()

    #Prepare data
    dataloader      =   load_locals(bs=bs,processor=None,local_dataset_path=ds_path,max_n=max_n)
    
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
            base_img    = torch.nn.functional.interpolate(item.to(DEV).type(torch.float),size=(final_dim,final_dim))

            #Apply distortion
            x           = apply_distortion(base_img,noise_sf=random.random()*max_strength,noise_iters=random.randint(0,max_distorts))    
            #Send thorugh model 
            pred        =  model.forward(x)

            #Loss
            loss        = loss_fn(pred,base_img)
            losses[-1].append(loss.mean().item())
            loss.backward()

            #Grad 
            optim.step()
            make_grid(x,pred,ep,batch_i)

        print(f"\tep loss={(sum(losses[-1])/len(losses[-1])):.5f}")
 

def hybrid_train(ds_path,n_ep,bs):

    #LOCAL OPTIONS
    start_dim           = 48 
    final_dim           = 128
    max_distorts        = 4
    max_strength        = .1
    max_n               = 512

    #TRAIN OPTIONS 
    lr                  = 1e-4
    wd                  = .001
    betas               = (.75,.9993)
    bs                  = bs


    #Create model and optim
    modelG              = UpSampleNet(start_dim=start_dim,final_dim=128).to(DEV)
    modelD              = DiscrNet(in_dim=final_dim).to(DEV)
    optimG              = torch.optim.Adam(modelG.parameters(),lr=lr,weight_decay=wd,betas=betas)
    optimD              = torch.optim.Adam(modelD.parameters(),lr=lr,weight_decay=wd,betas=betas)
    loss_fnG            = torch.nn.MSELoss()
    loss_fnD            = torch.nn.BCELoss()

    #Prepare data
    dataloader      =   load_locals(bs=bs,processor=None,local_dataset_path=ds_path,max_n=max_n)
    
    #Stats
    losses              = {'realLoss':[],'upscLoss':[],'simLoss':[]}

    for ep in range(n_ep):
        
        print(f"\n\tEPOCH {ep}")

        #Run train loop
        for batch_i,item in enumerate(dataloader):

            #Get base imgs and distorted_imgs
            base_imgs   = torch.nn.functional.interpolate(item.to(DEV).type(torch.float),size=(final_dim,final_dim))
            dist_imgs   = apply_distortion(base_imgs,noise_sf=random.random()*max_strength,noise_iters=random.randint(0,max_distorts))    

            #Train discriminator on base_imgs 
            optimD.zero_grad()
            true_prob   = modelD.forward(base_imgs.detach())
            true_labels = torch.ones(size=true_prob.shape,device=DEV,dtype=torch.float)
            realLoss    = loss_fnD(true_prob,true_labels)
            losses['realLoss'].append(realLoss.cpu().mean().item())
            #optimD.step()

            #Train discriminator on upsc_imgs
            upsc_imgs   = modelG.forward(dist_imgs)
            upsc_prob   = modelD.forward(upsc_imgs.detach())
            upsc_labels = torch.zeros(size=upsc_prob.shape,device=DEV,dtype=torch.float)
            upscLoss    = loss_fnD(upsc_prob,upsc_labels)
            losses['upscLoss'].append(upscLoss.cpu().mean().item())
            
            upscLoss.backward()
            realLoss.backward()
            optimD.step()

            #Train generator to fool
            #optimD.zero_grad()
            optimG.zero_grad()
            upsc_prob2  = modelD.forward(upsc_imgs)
            upscLossG   = loss_fnD(upsc_prob2,true_labels)
            upscLossG.backward()


            #Loss from similarity
            upsc_imgs   = modelG.forward(dist_imgs)
            simLoss     = loss_fnG(upsc_imgs,base_imgs)
            simLoss.backward()
            losses['simLoss'].append(simLoss.cpu().mean().item())
            optimG.step()

            #Check work
            make_grid(dist_imgs,upsc_imgs.detach(),ep,batch_i)

        print(f"\tsimLoss={(sum(losses['simLoss'])/len(losses['simLoss'])):.5f}\trealLoss={(sum(losses['realLoss'])/len(losses['realLoss'])):.5f}\tupscLoss={(sum(losses['upscLoss'])/len(losses['upscLoss'])):.5f}")
            
           

if __name__ == '__main__':

    hybrid_train("C:/data/images/converted_tensors/",10,16) 
        