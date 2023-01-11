import torch 
from torch.utils.data import Dataset, DataLoader
from networks import AudioGenerator,AudioGenerator2,AudioDiscriminator
import numpy 
import time 
import json
import os 
from dataset import reconstruct, upscale
import random 
import sys 
import pprint 
from utilities import weights_init, config_explorer, lookup, G_CONFIGS, D_CONFIGS, print_epoch_header,model_size
from sandboxG import build_gen,build_gen2, build_short_gen
from generatordev import build_encdec
class AudioDataSet(Dataset):

    def __init__(self,fnames,out_len):

        #Load files as torch tensors 
        self.data = []
        for file in fnames:
            arr = numpy.load(file,allow_pickle=True)
            arr = torch.from_numpy(arr).type(torch.float)
            if not arr.size()[-1] == out_len:
                input(f"{file} is bad w size {arr.size()}")
            self.data.append([arr,1])

        print(f"loaded {self.__len__()} samples")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        x = self.data[i][0]
        y = self.data[i][1]
        return x,y



class Trainer:

    def __init__(self,device:torch.DeviceObjType,ncz:int,outsize:int):

        #Create models
        self.Generator      = None 
        self.Discriminator  = None 

        self.device         = device
        self.ncz            = ncz
        self.outsize        = outsize
    
    #Import a list of configs for D and G in JSON format.
    def import_configs(self,config_filename:str):

        #Check if file exists
        if not os.path.exists(config_filename):
            print(f"Config file {config_filename} does not exist")
            return

        #Open file and read to JSON 
        file_contents       = open(config_filename,"r").read()
        try:
            config_dictionary   = json.loads(file_contents)

            #Ensure proper format 
            if not "G" in config_dictionary or not "D" in config_dictionary:
                print("Incorrectly formatted config file: Must contain G and D entries")
                return 
            
            self.config_file = config_dictionary

        except json.JSONDecodeError:
            print(f"file not in correct JSON format")
            return 

    #Create models from scratch using a config
    def create_models(self,D_config:dict,G_config:dict,run_parallel=True):

        #Ensure Proper Config Files 
        if not "factors" in G_config:
            print("Invalid Generator config, must contain Factor settings")
            exit(-1) 
        if not "channels" in G_config:
            print("Invalid Generator config, must contain Channel settings")
            exit(-1) 
        if not "scales" in G_config:
            print("Invalid Generator config, must contain Scale settings")
            exit(-1) 
       
        if not "kernels" in D_config:
            print("Invalid Discrimintator config, must contain Kernel settings")
            exit(-1) 
        if not "strides" in D_config:
            print("Invalid Discrimintator config, must contain Stride settings")
            exit(-1) 
        if not "paddings" in D_config:
            print("Invalid Discrimintator config, must contain Padding settings")
            exit(-1) 
        if not "channels" in D_config:
            print("Invalid Discrimintator config, must contain Channels settings")
            exit(-1)
        

        #Create Generator 
        self.Generator   = AudioGenerator2(         G_config['factors'],
                                                    G_config['channels'],
                                                    G_config['scales'])

        #Create Discriminator
        self.Discriminator   = AudioDiscriminator(  channels=D_config['channels'],
                                                    kernels=D_config['kernels'],
                                                    strides=D_config['strides'],
                                                    paddings=D_config['paddings'],
                                                    final_layer=D_config['final_layer'],
                                                    device=self.device,
                                                    verbose=False)

        #Init weights 
        self.Generator.apply(weights_init)
        self.Discriminator.apply(weights_init)

        #Check if mulitple GPUs 
        if torch.cuda.device_count() > 1 and run_parallel:
            print(f"Running model on {torch.cuda.device_count()} distributed GPUs")
            self.Generator      = torch.nn.DataParallel(self.Generator,device_ids=[id for id in range(torch.cuda.device_count())])
            self.Discriminator  = torch.nn.DataParallel(self.Discriminator,device_ids=[id for id in range(torch.cuda.device_count())])

        #Put both models on correct device 
        self.Generator      = self.Generator.to(self.device)
        self.Discriminator  = self.Discriminator.to(self.device) 
        #Save model config for later 
        self.G_config = G_config
        self.D_config = D_config

    #Import saved models 
    def load_models(self,D_config:dict,G_config:dict,D_params_fname:str,G_params_fname:str):
        
        #Create models
        self.create_models(D_config,G_config)

        #Load params
        self.Generator.load_state_dict(     torch.load(G_params_fname))
        self.Discriminator.load_state_dict( torch.load(D_params_fname))

    #Save state dicts and model configs
    def save_model_states(self,path:str,D_name="Discriminator_1",G_name="Generator_1"):
        
        #Ensure path is good 
        if not os.path.exists(path):
            os.mkdir(path)

        #Save model params to file 
        torch.save(     self.Generator.state_dict(),        os.path.join(path,G_name))
        torch.save(     self.Discriminator.state_dict(),    os.path.join(path,D_name)) 
        
        #Save configs
        with open(os.path.join(path,f"{G_name}_config"),"w") as config_file:
            config_file.write(json.dumps(self.G_config))
            config_file.close()
        with open(os.path.join(path,f"{D_name}_config"),"w") as config_file:
            config_file.write(json.dumps(self.D_config))
            config_file.close()

    #Get incoming data into a dataloader
    def build_dataset(self,filenames:list,batch_size:int,shuffle:bool,num_workers:int):

        #Save parameters 
        self.batch_size     = batch_size

        #Create sets
        self.dataset        = AudioDataSet(filenames,self.outsize)
        self.dataloader     = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

    #Set optimizers and error function for models
    def set_learners(self,D_optim,G_optim,error_fn):
        self.D_optim    = D_optim
        self.G_optim    = G_optim
        self.error_fn   = error_fn

    #Train models with NO accumulation
    def train(self,verbose=True):
        #Telemetry
        if verbose:
            width       = 100
            num_equals  = 50
            indent      = 4 
            n_batches   = len(self.dataset) / self.batch_size 
            t_init      = time.time()
            printed     = 0
            t_d         = [0] 
            t_g         = [0] 
            t_op_d      = [0]
            t_op_g      = [0]    
            d_fake      = 0 
            d_fake2     = 0 
            d_real      = 0
            d_random    = 0 

            print(" "*indent,end='')
            prefix = f"{int(n_batches)} batches  Progress" 
            print(f"{prefix}",end='')
            print(" "*(width-indent-len(prefix)-num_equals-2),end='')
            print("[",end='')
        
        d_error_fake = 0 
        d_error_real = 0 
        g_error = 0
        #Run all batches
        for i, data in enumerate(self.dataloader,0):

            if verbose:
                percent = i / n_batches
                while (printed / num_equals) < percent:
                    print("-",end='',flush=True)
                    printed+=1



            #####################################################################
            #                           TRAIN REAL                              #
            #####################################################################
            
            #Zero First
            self.Discriminator.zero_grad()

            #Prep real values
            t0                  = time.time()
            x_set               = data[0].to(self.device)
            x_len               = len(x_set)
            y_set               = torch.ones(size=(x_len,),dtype=torch.float,device=self.device)
            data_l              = time.time() - t0
            
            #Classify real set
            t_0 = time.time()
            real_class          = self.Discriminator.forward(x_set).view(-1)
            d_real              += real_class.mean().item()
            t_d[-1]             += time.time()-t_0

            #Calc error
            t0                  = time.time()
            d_error_real        = self.error_fn(real_class,y_set)
            t_op_d[-1]          += time.time() - t0
            
            #Back Propogate
            t_0                 = time.time()
            d_error_real.backward()
            t_op_d[-1]          += time.time()-t_0

            #####################################################################
            #                           TRAIN FAKE                              #
            #####################################################################
            
            #Generate samples
            t_0 = time.time()
            random_inputs           = torch.randn(size=(x_len,1,self.ncz),dtype=torch.float,device=self.device)
            generator_outputs       = self.Generator(random_inputs)
            t_g[-1] += time.time()-t_0

            #print(f"G out shape {random_inputs.shape}")

            #Ask Discriminator to classify fake samples 
            t_0 = time.time()
            fake_labels             = torch.zeros(size=(x_len,),dtype=torch.float,device=self.device)
            fake_class              = self.Discriminator.forward(generator_outputs.detach()).view(-1)
            d_fake                  += fake_class.mean().item()
            t_d[-1] += time.time()-t_0

            #Calc error
            t_0 = time.time()
            d_error_fake            = self.error_fn(fake_class,fake_labels)

            #Back Propogate
            d_error_fake.backward()
            self.D_optim.step()           
            t_op_d[-1] += time.time()-t_0

            #####################################################################
            #                           TRAIN GENR                              #
            #####################################################################

            self.Generator.zero_grad()

            #Classify the fakes again after Discriminator got updated 
            t_0 = time.time()
            fake_class2             = self.Discriminator.forward(generator_outputs).view(-1)
            d_fake2                 += fake_class2.mean().item()
            t_d[-1] += time.time()-t_0
            
            #Classify random vector 
            random_vect             = torch.randn(size=(1,2,self.outsize),dtype=torch.float,device=self.device)
            with torch.no_grad():
                d_random            += self.Discriminator.forward(random_vect).cpu().detach().item()
            
            #Find the error between the fake batch and real set  
            t_0 = time.time()
            g_error                 = self.error_fn(fake_class2,y_set)
            t_op_g[-1] += time.time()-t_0
            
            #Back Propogate
            t_0 = time.time()
            g_error.backward()   
            self.G_optim.step()
            t_op_g[-1] += time.time()-t_0

        
        if verbose:
            percent = (i+1) / n_batches
            while (printed / num_equals) < percent:
                print("-",end='',flush=True)
                printed+=1

        print(f"]")
        print("\n")
        out_1 = f"G forw={sum(t_d):.3f}s    G forw={sum(t_g):.3f}s    D back={sum(t_op_d):.3f}s    G back={sum(t_op_g):.3f}s    tot = {(time.time()-t_init):.2f}s"
        print(" "*(width-len(out_1)),end='')
        print(out_1,flush=True)
        out_2 = f"data_ld={(data_l):.3f}s    D(real)={(d_real/n_batches):.3f}    D(gen1)={(d_fake/n_batches):.4f}    D(rand)={(d_random/n_batches):.3f}"
        print(" "*(width-len(out_2)),end='')
        print(out_2)
        print("\n\n")
        t_d.append(0)
        t_g.append(0)

    #Train models with accumulation - DUBIOUS
    def train_accumulated(self,verbose=True,accu_bs=16):
        #Telemetry
        if verbose:
            accu_n_bat  = len(self.dataset) / accu_bs
            width       = 100
            num_equals  = 50
            indent      = 4 
            n_batches   = len(self.dataset) / self.batch_size 
            t_init      = time.time()
            printed     = 0
            t_d         = [0] 
            t_g         = [0] 
            t_op_d      = [0]
            t_op_g      = [0]    
            d_fake      = 0 
            d_fake2     = 0 
            d_real      = 0
            print(" "*indent,end='')
            prefix = f"{int(n_batches)} batches  Progress" 
            print(f"{prefix}",end='')
            print(" "*(width-indent-len(prefix)-num_equals-2),end='')
            print("[",end='')
        
        d_error_fake = 0 
        d_error_real = 0 
        g_error = 0
        #Run all batches
        for i, data in enumerate(self.dataloader,0):

            if verbose:
                percent = i / n_batches
                while (printed / num_equals) < percent:
                    print("-",end='',flush=True)
                    printed+=1



            #####################################################################
            #                           TRAIN REAL                              #
            #####################################################################

            #Prep real values
            t0                      = time.time()
            x_set                   = data[0].to(self.device)
            x_len                   = len(x_set)
            y_set                   = torch.ones(size=(x_len,),dtype=torch.float,device=self.device)
            data_l                  = time.time() - t0
            
            #Classify real set
            t_0 = time.time()
            real_class              = self.Discriminator.forward(x_set).view(-1)
            d_real                  += float(real_class.mean().item())
            t_d[-1] += time.time()-t_0

            #Calc error
            t0                      = time.time()
            d_error_real            = self.error_fn(real_class,y_set)
            d_error_real.backward()
            t_op_d[-1]              += time.time() - t0
            
            #####################################################################
            #                           TRAIN FAKE                              #
            #####################################################################
            
            #Generate samples
            t_0                     = time.time()
            random_inputs           = torch.randn(size=(x_len,self.input_channels,1),dtype=torch.float,device=self.device)
            generator_outputs       = self.Generator.forward(random_inputs)
            t_g[-1]                 += time.time()-t_0

            #Ask Discriminator to classify fake samples 
            t_0                     = time.time()
            fake_labels             = torch.zeros(size=(x_len,),dtype=torch.float,device=self.device)
            fake_class              = self.Discriminator.forward(generator_outputs.detach()).view(-1)
            d_fake                  += fake_class.mean().item()
            t_d[-1]                 += time.time()-t_0

            #Calc error
            t_0 = time.time()
            d_error_fake            = self.error_fn(fake_class,fake_labels)
            d_error_fake.backward()
            t_op_d[-1]              += time.time()-t_0

            #Back Propogate
            if((i*self.batch_size+1) % accu_bs) == 0 or ((i+1) == len(self.dataset)):
                t_0                 = time.time()
                self.D_optim.step()           
                t_op_d[-1]          += time.time()-t_0 
            #####################################################################
            #                           TRAIN GENR                              #
            #####################################################################

            #Classify the fakes again after Discriminator got updated 
            t_0 = time.time()
            fake_class2             = self.Discriminator.forward(generator_outputs).view(-1)
            d_fake2                 += fake_class2.mean().item()
            t_d[-1] += time.time()-t_0
            
            #Find the error between the fake batch and real set  
            t_0                     = time.time()
            g_error                 = self.error_fn(fake_class2,y_set)
            g_error.backward()
            t_op_g[-1]              += time.time()-t_0
            
            #Back Propogate
            if ((i*self.batch_size+1) % accu_bs) == 0 or ((i+1) == len(self.dataset)):
                t_0 = time.time()
                self.G_optim.step()
                self.Generator.zero_grad()
                self.Discriminator.zero_grad()
                t_op_g[-1]          += time.time()-t_0
        
        if verbose:
            percent = (i+1) / n_batches
            while (printed / num_equals) < percent:
                print("-",end='',flush=True)
                printed+=1

        print(f"]")
        print("\n")
        out_1 = f"G forw={sum(t_d):.3f}s    G forw={sum(t_g):.3f}s    D back={sum(t_op_d):.3f}s    G back={sum(t_op_g):.3f}s    tot = {(time.time()-t_init):.2f}s"
        print(" "*(width-len(out_1)),end='')
        print(out_1,flush=True)
        out_2 = f"data_ld={(data_l):.3f}s    D(real)={(d_real/n_batches):.3f}    D(fke)={(d_fake/n_batches):.4f}    D(gen)={(d_fake2/n_batches):.3f}"
        print(" "*(width-len(out_2)),end='')
        print(out_2)
        print("\n\n")
        t_d.append(0)
        t_g.append(0)

    #Get a sample from Generator
    def sample(self,out_file_path,sf=1):
        inputs = torch.randn(size=(1,1,self.ncz),dtype=torch.float,device=self.device)
        outputs = self.Generator.forward(inputs)
        outputs = outputs[0].cpu().detach().numpy()
        if sf > 1:
            outputs = upscale(outputs,sf)
        reconstruct(outputs,out_file_path)
        print(f"saved audio to {out_file_path}")

    #Train easier
    def c_exec(self,load,epochs,bs,D,G,optim_d,optim_g,filenames,ncz,outsize,sample_out,sf,verbose=False):
        self.outsize        =outsize
        self.ncz            = ncz
        self.Discriminator  = D 
        self.Generator      = G

        self.set_learners(optim_d,optim_g,torch.nn.BCELoss())

        for e in range(epochs):
            filenames   = random.sample(filenames,load)
            self.build_dataset(filenames,bs,True,2)
            if verbose:
                print_epoch_header(e,epochs)
                self.train(verbose=verbose)
        
            if (e+1) % 4 == 0:
                self.sample(f"{sample_out}_{e+1}.wav",sf=sf)

if __name__ == "__main__" and True:

    load    = 256 
    ep      = 32
    dev     = torch.device('cuda')

    
    D       = AudioDiscriminator(channels=[2,300,250,200,32,1],kernels=[9,33,33,33,33],paddings=[4,4,4,4,4],strides=[7,5,5,4,4,3,3,3],final_layer=182).to(dev)
    #ins     = torch.randn(size=(1,2,529200),device=dev)
    #input(f"outs {D(ins).shape}")
    if "linux" in sys.platform:
        root    = "/media/steinshark/stor_lg/music/dataset/LOFI_sf5_t60"
    else:
        root    = "C:/data/music/dataset/LOFI_sf5_t60"
    
    files   = [os.path.join(root,f) for f in os.listdir(root)]

    outsize = 529200
    for ncz in [2016*3]:
        for leak in [.2]:
            for r_fc in [False]:
                for r_ch in [False]:
                    for ver in [1]:
                    #G   = build_gen(ncz,reverse_factors=r_fc,reverse_channels=r_ch,ver=ver)
                        for bs in [8]:
                            for beta in [(.5,.5)]:
                                for lrs in [(.0001,.0005)]:
                                    
                                    t       = Trainer(dev,ncz,outsize)
                                    G = build_encdec(in_factors=[2,2,2,3,3,3,7],enc_factors=[2,3,7],dec_factors=[3,5,5,7,7],bs=8)                                
                                    if torch.cuda.device_count() > 1:
                                        G = torch.nn.DataParallel(G,device_ids=[0,1,2])
                                        D = torch.nn.DataParallel(D,device_ids=[0,1,2])
                                        print("Models created as DataParalell\n\n")

                                    #Create optims 
                                    optim_g = torch.optim.Adam(G.parameters(),lrs[1],betas=(beta[1],.999))
                                    optim_d = torch.optim.Adam(D.parameters(),lrs[0],betas=(beta[0],.999))

                                    series_name = f"outputs/{ncz}_beta{r_ch}_lrs{lrs}_ver{ver}"
                                    if not os.path.exists("outputs"):
                                        os.mkdir("outputs")
                                    if not os.path.isdir(series_name):
                                        os.mkdir(series_name)
                                    
                                    t.c_exec(load,ep,bs,D,G,optim_d,optim_g,files,ncz,outsize,f"{series_name}/",5,verbose=True)
