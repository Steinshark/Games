from collections import OrderedDict
import torch 
from torch.utils.data import Dataset, DataLoader
from networks import AudioDiscriminator, AudioDiscriminator2, AudioGenerator2
import numpy 
import time 
import json
import os 
from dataset import reconstruct, upscale
import random 
import sys 
import pprint 
from utilities import weights_initG,weights_initD, config_explorer, lookup, G_CONFIGS, D_CONFIGS, print_epoch_header,model_size
import sandboxG2
import sandboxG
from generatordev import build_encdec
from torch.distributed import init_process_group,destroy_process_group
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.optim import Adam 
from hashlib import md5
from cleandata import audit, load_state, check_tensor
from matplotlib import pyplot as plt 
from torch.profiler import record_function


#Dataset that is from all data combined
class AudioDataSet(Dataset):

    def __init__(self,fnames,out_len,save_to_sc=False):
        print(f"DATA")
        print(f"creating dataset: {md5(str(fnames).encode()).hexdigest()[:10]}")
        
        #Load files as torch tensors 
        self.data = []
        
        #
        for f in fnames:

            arr = numpy.load(f)
            tensor = torch.from_numpy(arr).view(1,-1).type(torch.float32)

            #Tensor has to be sent back to CPU
            self.data.append([tensor,1])
        


        print(f"loaded {self.__len__()} / {len(fnames)} samples")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        x = self.data[i][0]
        y = self.data[i][1]
        return x,y
    
    def __repr__():
        return "ADS"


#Dataset that is from the clean dataset
class CleanAudioDataSet(Dataset):

    def __init__(self,fnames,size,threshold=.9):
        print(f"\n\nCreating dataset size {size}")
        n_eqs   = 50 
        printed     = 0
        print("          [",end="",flush=True)
        print("="*n_eqs,end="]\n",flush=True)
        print("          [",end="",flush=True)
        load_state(fname=f"class_models/model_0.0617")
        self.data   = {"x":[],"y":[]}

        for i,fname in enumerate(fnames):
            
            while printed <( 50 * (self.__len__() / size)):
                print("=",end='',flush=True)
                printed += 1

            arr_as_tensor    = torch.from_numpy(numpy.load(fname)).type(torch.float)

            if check_tensor(arr_as_tensor,threshold=threshold ): 
                self.data['x'].append(arr_as_tensor.to(torch.device('cpu')))
                self.data['y'].append(1.0)  


                

            if self.__len__() >= size:
                print(f"]\ncompiled dataset size {self.__len__()}\tquality rating - {(self.__len__()/i):.2f}\n\n")
                return 
        
        print(f"]\ncompiled dataset size {self.__len__()}\tquality rating - {(self.__len__()/i):.2f}\n\n")



    def __len__(self):
        return len(self.data['x'])


    def __getitem__(self,i):
        return self.data['x'][i], self.data['y'][i]


#Provides the mechanism to train the model out
class Trainer:

    def __init__(self,device:torch.DeviceObjType,ncz:int,outsize:int,mode="single-channel"):

        #Create models
        self.Generator      = None 
        self.Discriminator  = None 

        self.device         = device
        self.ncz            = ncz
        self.outsize        = outsize
        self.mode           = mode 

        self.plots          = []
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
        self.Generator.apply(weights_initD)
        self.Discriminator.apply(weights_initD)

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
    def load_models(self,D_params_fname:str,G_params_fname:str,save_loc=None):
        
        #Create models
        #self.create_models(D_config,G_config)

        #Load params
        self.Generator.load_state_dict(     torch.load(G_params_fname))
        self.Discriminator.load_state_dict( torch.load(D_params_fname))

        if save_loc:
            contents        = [int(f.split("_")[1].replace(".wav","")) for f in os.listdir(save_loc)]
            self.epoch_num  = max(contents)

    #Save state dicts and model configs
    def save_model_states(self,path:str,D_name="Discriminator_1",G_name="Generator_1"):
        
        #Ensure path is good 
        if not os.path.exists(path):
            os.mkdir(path)

        #Save model params to file 
        torch.save(     self.Generator.state_dict(),        os.path.join(path,G_name))
        torch.save(     self.Discriminator.state_dict(),    os.path.join(path,D_name)) 
        
        ##Save configs
        #with open(os.path.join(path,f"{G_name}_config"),"w") as config_file:
        #    config_file.write(json.dumps(self.G_config))
        #    config_file.close()

        #with open(os.path.join(path,f"{D_name}_config"),"w") as config_file:
        #    config_file.write(json.dumps(self.D_config))
        #    config_file.close()

    #Get incoming data into a dataloader
    def build_dataset(self,filenames:list,size:int,batch_size:int,shuffle:bool,num_workers:int):

        #Save parameters 
        self.batch_size     = batch_size

        #Create sets
        self.dataset        = AudioDataSet(filenames,size)
        self.dataloader     = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)

    #Set optimizers and error function for models
    def set_learners(self,D_optim,G_optim,error_fn):
        self.D_optim    = D_optim
        self.G_optim    = G_optim
        self.error_fn   = error_fn

    #Train models with NO accumulation
    def train(self,verbose=True,gen_train_iters=1,proto_optimizers=True,t_dload=0):
        t_start = time.time()
        
        if proto_optimizers:
            torch.backends.cudnn.benchmark = True


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

            #Keep track of which batch we are on 
            final_batch     = i == len(self.dataloader)-1

            if verbose:
                percent = i / n_batches
                while (printed / num_equals) < percent:
                    print("-",end='',flush=True)
                    printed+=1



            #####################################################################
            #                           TRAIN REAL                              #
            #####################################################################
            
            #Zero First
            # OLD IMPLEMENTATION: self.Discriminator.zero_grad()
            for param in self.Discriminator.parameters():
                param.grad = None

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
            d_error_real.backward()
            t_op_d[-1]          += time.time() - t0
            
            #####################################################################
            #                           TRAIN FAKE                              #
            #####################################################################
            
            #Generate samples
            t_0 = time.time()
            if self.mode == "single-channel":
                random_inputs           = torch.randn(size=(x_len,1,self.ncz),dtype=torch.float,device=self.device)    
            elif self.mode == "multi-channel":
                random_inputs           = torch.randn(size=(x_len,self.ncz,1),dtype=torch.float,device=self.device)
           
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

            #Zero Grads
            for param in self.Generator.parameters():
                param.grad = None

            #Classify the fakes again after Discriminator got updated 
            t_0 = time.time()
            fake_class2                 = self.Discriminator.forward(generator_outputs).view(-1)
            t_d[-1] += time.time()-t_0
            

            #Classify random vector for reference 
            if final_batch:
                random_vect             = torch.randn(size=(1,self.outsize[0],self.outsize[1]),dtype=torch.float,device=self.device)
                with torch.no_grad():
                    d_random            = self.Discriminator.forward(random_vect).cpu().detach().item()
            
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

        #TELEMETRY
        print(f"]")
        print("\n")
        out_1 = f"G forw={sum(t_d):.3f}s    G forw={sum(t_g):.3f}s    D back={sum(t_op_d):.3f}s    G back={sum(t_op_g):.3f}s    tot = {(time.time()-t_init):.2f}s"
        print(" "*(width-len(out_1)),end='')
        print(out_1,flush=True)
        out_2 = f"t_dload={(t_dload):.2f}s    D(real)={(d_real/n_batches):.3f}    D(gen1)={(d_fake/n_batches):.4f}    D(rand)={d_random:.3f}"

        print(" "*(width-len(out_2)),end='')
        print(out_2)
        
        out_3 = f"er_real={(d_error_real):.3f}     er_fke={(d_error_fake):.4f}    g_error={(g_error):.3f}"
        print(" "*(width-len(out_3)),end='')
        print(out_3)
        print("\n\n")
       
        t_d.append(0)
        t_g.append(0)
        if (d_error_real < .00001) and ((d_error_fake) < .00001):
            return True

    #Train models with NO accumulation
    def train_profiler(self,verbose=True,gen_train_iters=1,proto_optimizers=True,t_dload=0):
        
        with torch.profiler.profile(
            activities=[                torch.profiler.ProfilerActivity.CPU,                torch.profiler.ProfilerActivity.CUDA            ],
            record_shapes=True
        ) as profiler:
            with record_function('model_inference'):
                t_start = time.time()
                
                if proto_optimizers:
                    torch.backends.cudnn.benchmark = True


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

                    #Keep track of which batch we are on 
                    final_batch     = i == len(self.dataloader)-1

                    if verbose:
                        percent = i / n_batches
                        while (printed / num_equals) < percent:
                            print("-",end='',flush=True)
                            printed+=1



                    #####################################################################
                    #                           TRAIN REAL                              #
                    #####################################################################
                    
                    #Zero First
                    # OLD IMPLEMENTATION: self.Discriminator.zero_grad()
                    for param in self.Discriminator.parameters():
                        param.grad = None

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
                    d_error_real.backward()
                    t_op_d[-1]          += time.time() - t0
                    
                    #####################################################################
                    #                           TRAIN FAKE                              #
                    #####################################################################
                    
                    #Generate samples
                    t_0 = time.time()
                    if self.mode == "single-channel":
                        random_inputs           = torch.randn(size=(x_len,1,self.ncz),dtype=torch.float,device=self.device)    
                    elif self.mode == "multi-channel":
                        random_inputs           = torch.randn(size=(x_len,self.ncz,1),dtype=torch.float,device=self.device)
                
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

                    #OLD IMPLIMENTATION - self.Generator.zero_grad()
                    for param in self.Generator.parameters():
                        param.grad = None
                    #Generate samples - TRY USING ONLY PREVIOUS FROM TRAIN DISC ON FAKE 
                    #t_0 = time.time()
                    #if self.mode == "single-channel":
                    #    random_inputs           = torch.randn(size=(x_len,1,self.ncz),dtype=torch.float,device=self.device)  
        #
                    #elif self.mode == "multi-channel":
                    #    random_inputs           = torch.randn(size=(x_len,self.ncz,1),dtype=torch.float,device=self.device)
                    #generator_outputs       = self.Generator(random_inputs)
                    #t_g[-1] += time.time()-t_0
                    
                    #Classify the fakes again after Discriminator got updated 
                    t_0 = time.time()
                    fake_class2                 = self.Discriminator.forward(generator_outputs).view(-1)
                    t_d[-1] += time.time()-t_0
                    

                    if final_batch:
                        #Classify random vector 
                        random_vect             = torch.randn(size=(1,self.outsize[0],self.outsize[1]),dtype=torch.float,device=self.device)
                        with torch.no_grad():
                            d_random            = self.Discriminator.forward(random_vect).cpu().detach().item()
                    
                    #Find the error between the fake batch and real set  
                    t_0 = time.time()
                    g_error                 = self.error_fn(fake_class2,y_set)
                    t_op_g[-1] += time.time()-t_0
                    
                    #Back Propogate
                    t_0 = time.time()
                    g_error.backward()   
                    self.G_optim.step()
                    t_op_g[-1] += time.time()-t_0
                    profiler.step()
                    
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
                out_2 = f"t_dload={(t_dload):.2f}s    D(real)={(d_real/n_batches):.3f}    D(gen1)={(d_fake/n_batches):.4f}    D(rand)={d_random:.3f}"

                print(" "*(width-len(out_2)),end='')
                print(out_2)
                
                out_3 = f"er_real={(d_error_real):.3f}     er_fke={(d_error_fake):.4f}    g_error={(g_error):.3f}"
                print(" "*(width-len(out_3)),end='')
                print(out_3)
                print("\n\n")
            
                t_d.append(0)
                t_g.append(0)
                if (d_error_real < .00001) and ((d_error_fake) < .00001):
                    return True
            


        with open("TEL.txt","w") as f:
            profdata = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            f.write(profdata)
            f.close()

    #Train NOT the DCGAN way 
    def train_online(self,verbose,g_train_iters=1,d_train_iters=1,g_train_ratio=4):
        
        torch.backends.cudnn.benchmark = True


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

            #Keep track of which batch we are on 
            final_batch     = i == len(self.dataloader)-1

            if verbose:
                percent = i / n_batches
                while (printed / num_equals) < percent:
                    print("-",end='',flush=True)
                    printed+=1



            #####################################################################
            #                           TRAIN REAL                              #
            #####################################################################
            
            #Put discriminator in training mode
            self.Discriminator.train()

            #Zero D
            for p in self.Discriminator.parameters():
                p.grad = None 

            #Train on real set 
            real_lofi       = data[0].to(self.device)
            real_labels     = torch.ones(size=(bs,1,1),device=self.device)

            real_class      = self.Discriminator.forward(real_lofi)

            #Calc loss 
            d_err_r           = self.error_fn(real_class,real_labels)
            d_err_r.backward()

            #Put generator in non-training mode 
            self.Generator.eval()

            #Get fake batch
            if self.mode == "single-channel":
                    random_inputs           = torch.randn(size=(self.batch_size,1,self.ncz),dtype=torch.float,device=self.device)  
            elif self.mode == "multi-channel":
                random_inputs           = torch.randn(size=(self.batch_size,self.ncz,1),dtype=torch.float,device=self.device)
            with torch.no_grad():
                fake_lofi       = self.Generator(random_inputs)

            fake_class      = self.Discriminator(fake_lofi)
            fake_labels     = torch.zeros(size=(bs,1,1),device=self.device)
            
            #Add to loss 
            d_err_f           = self.error_fn(fake_class,fake_labels)

            d_err_f.backward()
            self.D_optim.step()
            

            #Switch to train generator
            for p in self.Generator:
                p.grad  = None 

            self.Discriminator.eval()
            self.Generator.train()  

            #Get fake batch
            if self.mode == "single-channel":
                    random_inputs           = torch.randn(size=(self.batch_size*g_train_ratio,1,self.ncz),dtype=torch.float,device=self.device)  
            elif self.mode == "multi-channel":
                random_inputs           = torch.randn(size=(self.batch_size*g_train_ratio,self.ncz,1),dtype=torch.float,device=self.device)
            fake_lofi_t       = self.Generator(random_inputs)

            fake_class_g    = self.Discriminator(fake_lofi_t)
            real_labels     = torch.ones(size=(bs*g_train_ratio,1,1),device=self.device)
            g_err           = self.error_fn(fake_class_g,real_labels)

            g_err.backward()
            self.G_optim.step()

        print(f"]")
        print("\n")
        out_1 = f"G forw={sum(t_d):.3f}s    G forw={sum(t_g):.3f}s    D back={sum(t_op_d):.3f}s    G back={sum(t_op_g):.3f}s    tot = {(time.time()-t_init):.2f}s"
        print(" "*(width-len(out_1)),end='')
        print(out_1,flush=True)
        out_2 = f"data_ld={(0):.3f}s    D(real)={real_class.cpu().mean():.3f}    D(fake)={fake_class.cpu().mean():.4f}    D(genr)={fake_class_g.cpu().mean():.3f}"

        print(" "*(width-len(out_2)),end='')
        print(out_2)
        
        out_3 = f"d_error={fake_class.cpu().mean():.4f}    g_error={(g_error):.3f}"
        print(" "*(width-len(out_3)),end='')
        print(out_3)
        print("\n\n")
        if (d_error_real < .00001) and ((d_error_fake) < .00001) and False:
            return True


    #Get a sample from Generator
    def sample(self,out_file_path,sf=1,sample_set=100,store_plot=True,store_path="plots"):
        best_score      = 0
        best_sample     = None 

        all_scores      = [] 
        #Search for best sample  
        with torch.no_grad():

            for _ in range(sample_set):
                
                #Create inputs  
                if self.mode == "multi-channel":
                    inputs  = torch.randn(size=(1,self.ncz,1),dtype=torch.float,device=self.device)
                elif self.mode == "single-channel":
                    inputs  = torch.randn(size=(1,1,self.ncz),dtype=torch.float,device=self.device)
                
                #Grab score
                outputs     = self.Generator.forward(inputs)
                score       = self.Discriminator(outputs)

                #Check best score 
                val         = score.item()
                #Check if better was found 
                if val > best_score:
                    best_score      = val 
                    best_sample     = outputs.view(1,-1).cpu().detach().numpy()
                
                all_scores.append(val)
        
        self.plots.append(sorted(all_scores))
        #Telemetry 
        print(f"average was {sum(all_scores)/len(all_scores):.3f} vs. best  was {best_score:.3f}")

        #Store plot 
        if store_plot:
            if not os.path.isdir(store_path):
                os.mkdir(store_path)

            all_scores = sorted(all_scores)

            #Plot current, 3 ago, 10 ago 
            plt.plot(list(range(sample_set)),all_scores,color="dodgerblue",label="current")
            try:
                if self.epoch_num >= 3:
                    plt.plot(list(range(sample_set)),self.plots[self.epoch_num-3],color="darkorange",label="prev-3")
                if self.epoch_num >= 10:
                    plt.plot(list(range(sample_set)),self.plots[self.epoch_num-10],color="crimson",label="prev-10")
            except IndexError:
                pass 
            plt.legend()
            plt.savefig(os.path.join(store_path,os.path.basename(out_file_path).replace(".wav","")))
            plt.cla()
        #Bring up to 2 channels 
        if self.outsize[0] == 1:
            outputs = numpy.array([best_sample[0],best_sample[0]]) 
        
        #Rescale up 
        if sf > 1:
            outputs = upscale(outputs,sf)
        reconstruct(outputs,out_file_path)
        print(f"saved audio to {out_file_path}")

    #Train easier
    def c_exec(self,load,epochs,bs,optim_d,optim_g,filenames,ncz,outsize,sample_out,sf,verbose=False,sample_rate=1):
        self.outsize        =outsize
        self.ncz            = ncz

        self.set_learners(optim_d,optim_g,torch.nn.BCELoss())

        epochs = self.epoch_num+epochs
        for e in range(self.epoch_num,epochs):
            self.epoch_num      = e 
            t0 = time.time()
            train_set   = random.sample(filenames,load)
            self.build_dataset(train_set,load,bs,True,4)
            if verbose:
                print_epoch_header(e,epochs)
            failed = self.train(verbose=verbose,t_dload=time.time()-t0)#,gen_train_iters=gen_train_iters,proto_optimizers=proto_optimizers)
            if (e+1) % sample_rate == 0:
                self.sample(f"{sample_out}_{e+1}.wav",sf=sf)
            if failed:
                return 
            self.save_model_states("models","D_model","G_model")


#Training Section
if __name__ == "__main__" and True:

    bs          = 8
    ep          = 64
    dev         = torch.device('cuda')
    wd          = .00002
    kernels     = [5,17,65,33,33,17,9]
    paddings    = [int(k/2) for k in kernels]
    load        = eval(sys.argv[sys.argv.index("ld")+1]) if "ld" in sys.argv else bs*250
    outsize     = (1,int(529200/3))



    if "linux" in sys.platform:
        root    = "/media/steinshark/stor_lg/music/dataset/LOFI_sf5_t60"
    else:
        root    = "C:/data/music/dataset/LOFI_sf5_t20_thrsh.9"
    
    files   = [os.path.join(root,f) for f in os.listdir(root)]

    

    # for ch_i,ch in enumerate([[200,150,100,50,25,2]]):
    #     for k_i,ker in enumerate([[1001,501,201,33,33,17]]):
    configs = { 
                #"upsample" : {"init":sandboxG.build_upsamp,"beta":(.5,.5),"lrs":(.0002,.0002),"ncz":2048,"kernel":0,"factor":0,"leak":.05}
                #"TEST" : {"init":sandboxG.buildBest,"beta": (.5,.5),"lrs": (.0001,.0002), "ncz":1024,"kernel":0,"factor":0,"leak":.2},
                #"best_SGD_f0"  : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00004,.00006), "ncz":512,"kernel":0,"factor":0,"leak":.2},
                #"best_SGD_f1_1024"  : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.0001,.0002), "ncz":512,"kernel":0,"factor":1,"leak":.2},
                "best_SGD"   : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00004,.00006), "ncz":256,"kernel":0,"factor":0,"leak":.2},
                #"best_SGD_f1_512"   : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.0001,.0002), "ncz":256,"kernel":0,"factor":1,"leak":.2},
                
                #"conf_2" : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00002,.0002), "ncz":1024,"kernel":1,"factor":0,"leak":.2}
                #"conf_2" : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00002,.00002), "ncz":128,"kernel":1,"factor":0,"leak":.2},
                #"conf_3" : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00002,.00002), "ncz":1024,"kernel":2,"factor":0,"leak":.2},

                #"conf_4" : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00002,.00002), "ncz":64,"kernel":0,"factor":1,"leak":.2},
                #"conf_5" : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00002,.00002), "ncz":128,"kernel":1,"factor":1,"leak":.2},
                #"conf_6" : {"init":sandboxG.buildBestMod1,"beta": (.5,.5),"lrs": (.00002,.00002), "ncz":1024,"kernel":2,"factor":1,"leak":.2},

                #"conf_7" : {"init":sandboxG.buildBestMod1,"beta": (.2,.2),"lrs": (.00002,.00002), "ncz":64,"kernel":0,"factor":0,"leak":.2},
                #"conf_8" : {"init":sandboxG.buildBestMod1,"beta": (.2,.2),"lrs": (.00002,.00002), "ncz":128,"kernel":1,"factor":0,"leak":.2},
                #"conf_9" : {"init":sandboxG.buildBestMod1,"beta": (.2,.2),"lrs": (.00002,.00002), "ncz":1024,"kernel":2,"factor":0,"leak":.2},
                

    }

    for config in configs:
        
        name                = config 
        ncz                 = configs[config]['ncz'] 
        lrs                 = configs[config]['lrs']
        betas               = configs[config]['beta']
        kernel              = configs[config]['kernel']
        factor              = configs[config]['factor']
        leak                = configs[config]['leak']
        series_name = f"outputs/{name}_ncz{ncz}_lr{lrs}_beta{betas}_kern{kernel}_fact{factor}_l{leak}"
        loading             = True if not "lf" in sys.argv else eval(sys.argv[sys.argv.index("lf")+1]) 
        
        #Build Trainer 
        t                   = Trainer(torch.device('cuda'),ncz,outsize,mode="multi-channel")

        #Create Generator and Discriminator
        G: torch.nn.Module
        D: torch.nn.Module
        G                   = configs[config]["init"](ncz=ncz,out_ch=1,leak=leak,kernel_ver=kernel,factor_ver=factor) 
        G.apply(weights_initD)

        #Try new D 
        from cleandata import _D
        D                   = _D
        D.apply(weights_initD)

        t.Discriminator     = D 
        t.Generator         = G

        #Check output sizes are kosher 
        inpv2               = torch.randn(size=(1,ncz,1),device=torch.device("cuda"),dtype=torch.float)
        print(f"MODELS:")
        if loading:
            t.load_models("models/D_model","models/G_model",save_loc=series_name)
            print("loaded models")
        else:
            t.epoch_num     = 0
        print(f"G stats:\tout  - {G.forward(inpv2).shape}\n        \tsize - {(sum([p.nelement()*p.element_size() for p in G.parameters()])/1000000):.2f}MB")
        print(f"D stats:\tout  - {D(G.forward(inpv2)).shape}\n        \tsize - {(sum([p.nelement()*p.element_size() for p in D.parameters()])/1000000):.2f}MB\n        \tval  - {D(G.forward(inpv2)).detach().cpu().item():.3f}")

        #Create optims 
        #optim_d = torch.optim.AdamW(D.parameters(),lrs[0],betas=(betas[0],.999))
        #optim_g = torch.optim.AdamW(G.parameters(),lrs[1],betas=(betas[1],.999))
        
        optim_d = torch.optim.SGD(D.parameters(),lrs[0],momentum=.9,weight_decay=lrs[0]/10)
        optim_g = torch.optim.SGD(G.parameters(),lrs[1],momentum=.9,weight_decay=lrs[1]/10)
        
        
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        if not os.path.isdir(series_name):
            os.mkdir(series_name)
        
        t.c_exec(load,ep,bs,optim_d,optim_g,files,ncz,outsize,f"{series_name}/",5,verbose=True,sample_rate=1)
