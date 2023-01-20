import torch 
from torch.optim import Adam,AdamW,Adamax,Adadelta,Adagrad,SGD
from torch.distributed import init_process_group,destroy_process_group
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import Dataset,DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  

import numpy 
import random 
import os 
import time 

import sandboxG2
import sandboxG

from SoundBooth import AudioDataSet
from utilities import weights_initG,weights_initD, config_explorer, lookup, G_CONFIGS, D_CONFIGS, print_epoch_header,model_size
from networks import AudioDiscriminator

#Track CONSTANTS 
N_GPUS = torch.cuda.device_count()
NCZ             = 2048
LOAD            = 512
EPOCHS          = 128
G               = sandboxG2.build_short_gen(NCZ)
D               = AudioDiscriminator(channels=[1,32,32,64,64,256,256,1],kernels=[15,11,9,9,5,5,5],strides=[12,9,7,7,5,5,4],paddings=[7,5,4,4,2,2,2],device=torch.device('cuda'),final_layer=1,verbose=False)


#HYPER PARAMS 
BS              = 8 
LR_D            = .0001
LR_G            = .0002
BETA_1          = .75 

#Init the process
def start_process(device_id,device_count):

    os.environ['MASTER_ADDR']        = '10.0.0.60'
    os.environ["MASTER_PORT"]      = "8881"

    init_process_group("gloo",rank=device_id,world_size=device_count)


#Destroy process 
def end_process():
    destroy_process_group()


#Start model to get ready 
def prep_data(device_id,device_count,bs,fnames):
    dataset     = AudioDataSet(fnames,(1,529200))
    sampler     = DistributedSampler(dataset,num_replicas=device_count,rank=device_id,shuffle=False,drop_last=True)
    dataloader  = DataLoader(dataset,batch_size=bs,pin_memory=True,num_workers=2,shuffle=False,sampler=sampler,drop_last=True)

    return dataloader


def run(device_id,device_count):
    
    start_process(device_id,device_count)

    dataloader  = prep_data(device_id,device_count,BS,random.sample([os.path.join("C:/data/music/dataset/LOFI_sf5_t60",f) for f in os.listdir("C:/data/music/dataset/LOFI_sf5_t60")],LOAD)) 

    G               = sandboxG2.build_short_gen(NCZ)
    D               = AudioDiscriminator(channels=[1,32,32,64,64,256,256,1],kernels=[15,11,9,9,5,5,5],strides=[12,9,7,7,5,5,4],paddings=[7,5,4,4,2,2,2],device=torch.device('cuda'),final_layer=1,verbose=False)

    G.to(device_id)
    D.to(device_id)

    G               = DDP(G,device_ids=[device_id],output_device=device_id,find_unused_parameters=True)
    D               = DDP(D,device_ids=[device_id],output_device=device_id,find_unused_parameters=True)


    G_optim         = Adam(G.parameters(),betas=(BETA_1,.999),lr=LR_G)
    D_optim         = Adam(D.parameters(),betas=(BETA_1,.999),lr=LR_D)

    loss_fn         = torch.nn.BCELoss()

    t_0             = time.time()
    for ep in range(EPOCHS):

        dataloader.sampler.set_epoch(ep)

        for setp, x in enumerate(dataloader):
            

            ###############################################################################
            #                           Train Discriminator 
            ###############################################################################
            #ZERO D
            for p in D.parameters():
                p.grad  = None 

            #Train real
            D.train()
            real_labels     = torch.ones(size=(BS,1,1),device=torch.device(device_id))
            real_class      = D(x)
            real_err        = loss_fn(real_class,real_labels)
            real_err.backward()

            #Train fake 
            fake_labels     = torch.zeros(size=(BS,1,1),device=torch.device(device_id))
            G.eval()
            with torch.no_grad():
                fake_lofi       = G(torch.randn(size=(BS,1,529200),device=torch.device(device_id)))
            fake_class      = D(fake_lofi)
            fake_err        = loss_fn(fake_class,fake_labels)
            fake_err.backward()

            #Step Grads 
            D_optim.step()

            ###############################################################################
            #                           Train Generator 
            ###############################################################################

            #Switch modes 
            D.eval()
            G.train()

            #Zero G Grad 
            for p in G.parameters():
                p.grad  = None 
            
            #Generate music 
            fake_lofi_2     = G(torch.randn(size=(BS,1,529200),device=torch.device(device_id)))
            fake_class_g    = D(fake_lofi_2)

            #Get err 
            fake_err_g      = loss_fn(fake_class_g,torch.ones(size=(BS,1,1),device=torch.device(device_id)))
            fake_err_g.backward()
            
            G_optim.step()

        print(f"EPOCH[{ep}\t]:{device_id} - d_loss={((real_err+fake_err)/2):.4f} g_loss={fake_err_g:.4f} | d_real={real_class.mean():.32f} d_fake={fake_class.mean():.3f} d_gen={fake_class_g.mean():.3f} {(time.time()-t_0):.1f}s")
        t_0     = time.time()
    end_process()



if __name__ == "__main__":

    from torch import multiprocessing as torchMP

    torchMP.spawn(run,args=[N_GPUS],nprocs=N_GPUS)