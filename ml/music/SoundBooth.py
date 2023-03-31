import torch 
from collections import OrderedDict
from networks import AudioDiscriminator, AudioDiscriminator2, AudioGenerator2
from torch.utils.data import Dataset, DataLoader
import numpy 
import time 
import json
import os 
from dataset import reconstruct, upscale, normalize_peak
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

    def __init__(self,fnames,out_len,save_to_sc=False,normalizing=1):
        print(f"DATA")
        print(f"creating dataset: {md5(str(fnames).encode()).hexdigest()[:10]}\tnormalizing to {normalizing}")
        
        #Load files as torch tensors 
        self.data = []
        
        #
        for f in fnames:

            arr = numpy.load(f)
            if normalizing:
                arr = (normalize_peak(arr,peak=normalizing))
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
        self.training_errors  = {"g_err":     [],
                               "d_err_real":[],
                               "d_err_fake":[]}

        self.training_classes = {"g_class":[],
                                 "d_class":[],
                                 "r_class":[]}
    
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
    def load_models(self,D_params_fname:str,G_params_fname:str,ep_start=None):
        
        #Create models
        #self.create_models(D_config,G_config)

        #Load params
        self.Generator.load_state_dict(     torch.load(G_params_fname))
        self.Discriminator.load_state_dict( torch.load(D_params_fname))

        if ep_start:
            self.epoch_num = ep_start

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
        self.dataset        = AudioDataSet(filenames,size,normalizing=False)
        self.dataloader     = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)

    #Set optimizers and error function for models
    def set_learners(self,D_optim,G_optim,error_fn):
        self.D_optim    = D_optim
        self.G_optim    = G_optim
        self.error_fn   = error_fn

    #Train models with NO accumulation
    def train_kbest(self,verbose=True,gen_train_iters=1,proto_optimizers=True,t_dload=0,k_best=10):
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
            #This train algorithm will generate n batches and pick the batch with the highest score to train on - def=10
            t_0 = time.time()
            if self.mode == "single-channel":
                best_z_vect           = torch.randn(size=(x_len,1,self.ncz),dtype=torch.float,device=self.device)    
            elif self.mode == "multi-channel":
                best_z_vect           = torch.randn(size=(x_len,self.ncz,1),dtype=torch.float,device=self.device)

            top_score       = 0
            for n in range(k_best):
                if self.mode == "single-channel":
                    random_inputs           = torch.randn(size=(x_len,1,self.ncz),dtype=torch.float,device=self.device)    
                elif self.mode == "multi-channel":
                    random_inputs           = torch.randn(size=(x_len,self.ncz,1),dtype=torch.float,device=self.device)
                with torch.no_grad():
                    generator_outputs           = self.Generator(random_inputs)
                    fake_score                  = self.Discriminator.forward(generator_outputs).view(-1).mean().item()

                    if fake_score > top_score:
                        best_z_vect = random_inputs 
                        top_score = fake_score 
            
            #Train for real 
            generator_outputs                   = self.Generator(best_z_vect)
            t_g[-1] += time.time()-t_0


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
            #Train for real 
            generator_outputs                   = self.Generator(best_z_vect)
            fake_class2                         = self.Discriminator(generator_outputs).view(-1)
            t_d[-1] += time.time()-t_0
            

            #Classify random vector for reference 
            if final_batch:
                random_vect             = torch.randn(size=(1,self.outsize[0],self.outsize[1]),dtype=torch.float,device=self.device)
                with torch.no_grad():
                    d_random            = self.Discriminator.forward(random_vect).cpu().detach().item()
            
            #Find the error between the fake batch and real set  
            t_0 = time.time()
            g_error                 = self.error_fn(fake_class2,torch.ones(size=(x_len,),dtype=torch.float,device=self.device))
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

        self.training_errors['d_err_real'].append(d_error_real.cpu().item())
        self.training_errors['d_err_fake'].append(d_error_fake.cpu().item())
        self.training_errors['g_err']     .append(g_error.cpu().item())

        self.training_classes['d_class'].append(d_real/n_batches)
        self.training_classes['g_class'].append(d_fake/n_batches)
        self.training_classes['r_class'].append(d_random)

        self.training_classes
        if (d_error_real < .0001) and ((d_error_fake) < .0001):
            return True

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

        self.training_errors['d_err_real'].append(d_error_real.cpu().item())
        self.training_errors['d_err_fake'].append(d_error_fake.cpu().item())
        self.training_errors['g_err']     .append(g_error.cpu().item())

        self.training_classes['d_class'].append(d_real/n_batches)
        self.training_classes['g_class'].append(d_fake/n_batches)
        self.training_classes['r_class'].append(d_random)
        if (d_error_real < .0001) and ((d_error_fake) < .0001):
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
    def sample(self,out_file_path,sf=1,sample_set=100,store_plot=True,store_file="plots"):
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
            plt.cla()
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
            plt.savefig(store_file)
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
    def c_exec(self,load,epochs,bs,optim_d,optim_g,filenames,ncz,outsize,series_path,sf,verbose=False,sample_rate=1):
        self.outsize        = outsize
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
            failed = self.train_kbest(verbose=verbose,t_dload=time.time()-t0,proto_optimizers=False,k_best=3)#,gen_train_iters=gen_train_iters,proto_optimizers=proto_optimizers)
            if (e+1) % sample_rate == 0:
                self.save_run(series_path)
            if failed:
                return 
            self.save_model_states("models","D_model","G_model")

    #Save telemetry and save to file 
    def save_run(self,series_path):
        
        #Create samples && plots 
        self.sample(os.path.join(series_path,'samples',f'run{self.epoch_num}.wav'),sf=5,store_file=os.path.join(series_path,'distros',f'run{self.epoch_num}'))

        #Create errors and classifications 
        plt.cla()
        fig,axs     = plt.subplots(nrows=2,ncols=1)
        fig.set_size_inches(20,10)
        axs[0].plot(list(range(len(self.training_errors['g_err']))),self.training_errors['g_err'],label="G_err",color="dodgerblue")
        axs[0].plot(list(range(len(self.training_errors['d_err_real']))),self.training_errors['d_err_real'],label="D_err_real",color="darkorange")
        axs[0].plot(list(range(len(self.training_errors['d_err_fake']))),self.training_errors['d_err_fake'],label="D_err_fake",color="goldenrod")
        axs[0].set_title("Model Loss vs Epoch")
        axs[0].set_xlabel("Epoch #")
        axs[0].set_ylabel("BCE Loss")
        axs[0].legend()
        axs[1].plot(list(range(len(self.training_classes['d_class']))),self.training_classes['d_class'],label="Real Class",color="darkorange")
        axs[1].plot(list(range(len(self.training_classes['g_class']))),self.training_classes['g_class'],label="Fake Class",color="dodgerblue")
        axs[1].plot(list(range(len(self.training_classes['r_class']))),self.training_classes['r_class'],label="Rand Class",color="dimgrey")
        axs[1].set_title("Model Classifications vs Epoch")
        axs[1].set_xlabel("Epoch #")
        axs[1].set_ylabel("Classification")
        axs[1].legend()
        fig.savefig(f"{os.path.join(series_path,'errors',f'Error and Classifications')}")
        plt.close()
        #Save models 
        self.save_model_states(os.path.join(series_path,'models'),D_name=f"D_run{self.epoch_num}",G_name=f"G_run{self.epoch_num}")

        #Save telemetry to file every step - will try to be recovered at beginning
        stash_dictionary    = json.dumps({"training_errors":self.training_errors,"training_classes":self.training_classes})
        with open(os.path.join(series_path,"data","data.txt"),"w") as save_file:
            save_file.write(stash_dictionary)
        save_file.close()

        
        




"""
Method for saving progress 
   1. Create Folder system: 
        - root 
            - samples 
            - errors 
            - distros  
            - models 
            - data

"""

#Training Section
if __name__ == "__main__" and True:

    bs          = 8
    ep          = 1000
    dev         = torch.device('cuda')
    kernels     = [5,17,65,33,33,17,9]
    paddings    = [int(k/2) for k in kernels]
    load        = eval(sys.argv[sys.argv.index("ld")+1]) if "ld" in sys.argv else bs*250
    outsize     = (1,int(529200/3))

    if not os.path.exists("model_runs"):
        os.mkdir(f"model_runs")


    if "linux" in sys.platform:
        root    = "/media/steinshark/stor_lg/music/dataset/LOFI_sf5_t60"
    else:
        root    = "C:/data/music/dataset/LOFI_sf5_t20_peak1_thrsh.95"
    
    files   = [os.path.join(root,f) for f in os.listdir(root)]

    

    # for ch_i,ch in enumerate([[200,150,100,50,25,2]]):
    #     for k_i,ker in enumerate([[1001,501,201,33,33,17]]):
    configs = { 
                "mod9_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.00002), "ncz":400,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":8},
                "mod8_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":256,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":8},
                "mod7_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":64,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":8},
                
                "mod1_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.000025), "ncz":64,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":4},
                "mod2_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.000025), "ncz":128,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":4},
                "mod3_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.000025), "ncz":256,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":4},

                "mod4_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.000025), "ncz":64,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":2},
                "mod5_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.000025), "ncz":128,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":2},
                "mod6_9"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.000025), "ncz":256,"kernel":0,"factor":0,"leak":.2,"momentum":.9,"bs":2},

                "mod1_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":64,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":2},
                "mod2_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":128,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":2},
                "mod3_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":256,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":2},

                "mod4_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":64,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":4},
                "mod5_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":128,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":4},
                "mod6_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":256,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":4},

                "mod7_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":64,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":8},
                "mod8_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":128,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":8},
                "mod9_7"   : {"init":sandboxG.buildBestMod1,"lrs": (.0001,.0001), "ncz":256,"kernel":0,"factor":0,"leak":.2,"momentum":.7,"bs":8},
                

    }

    configs = { "bigbatch"   : {"init":sandboxG.buildBestMod1,"lrs": (.00001,.000015), "ncz":200,"kernel":0,"factor":0,"leak":.5,"momentum":.91,"bs":10}}

    for config in configs:  

        #Load values 
        name                    = config 
        ncz                     = configs[config]['ncz'] 
        lrs                     = configs[config]['lrs']
        momentum                = configs[config]['momentum']
        kernel                  = configs[config]['kernel']
        factor                  = configs[config]['factor']
        leak                    = configs[config]['leak']
        bs                      = configs[config]['bs']
        series_path             = f"model_runs/{name}_ncz{ncz}_lr{lrs}_moment{momentum}_l{leak}"
        loading                 = True if not "lf" in sys.argv else eval(sys.argv[sys.argv.index("lf")+1]) 

        #Build Trainer 
        t                       = Trainer(torch.device('cuda'),ncz,outsize,mode="multi-channel")


        lrs                     = (.000001,.0000015)

        #Create folder system 
        if not os.path.exists(series_path):
            os.mkdir(series_path)
            for folders in ["samples","errors","distros","models","data"]:
                os.mkdir(os.path.join(series_path,folders))
        else:
            if not loading:
                break 
            stash_dict          = json.loads(open(os.path.join(series_path,"data","data.txt"),"r").read())
            t.training_errors   = stash_dict['training_errors']
            t.training_classes  = stash_dict['training_classes']
        #Try to load telemetry 



        #Create Generator and Discriminator
        G: torch.nn.Module
        D: torch.nn.Module
        G                       = configs[config]["init"](ncz=ncz,out_ch=1,leak=leak,kernel_ver=kernel,factor_ver=factor) 
        G.apply(weights_initD)

        #Try new D 
        from cleandata import _D
        D                       = _D
        D.apply(weights_initD)

        t.Discriminator         = D 
        t.Generator             = G

        #Check output sizes are kosher 
        inpv2                   = torch.randn(size=(1,ncz,1),device=torch.device("cuda"),dtype=torch.float)
        print(f"MODELS: {name}")
        if loading:
            root = os.path.join(series_path,"models")
            max_run = max([int(f.split("_run")[-1]) for f in os.listdir(os.path.join(series_path,"models"))])
            print(f"loading from epoch {root}")
            t.load_models(os.path.join(root,f"D_run{max_run}"),os.path.join(root,f"G_run{max_run}"),ep_start=max_run)
            print("loaded models")
        else:
            t.epoch_num     = 0
        print(f"G stats:\tout  - {   G.forward(inpv2).shape }\n        \tsize - {(sum([p.nelement()*p.element_size() for p in G.parameters()])/1000000):.2f}MB")
        print(f"D stats:\tout  - {D( G.forward(inpv2)).shape}\n        \tsize - {(sum([p.nelement()*p.element_size() for p in D.parameters()])/1000000):.2f}MB\n        \tval  - {D(G.forward(inpv2)).detach().cpu().item():.3f}")

        #Create optims 
        #optim_d = torch.optim.AdamW(D.parameters(),lrs[0],betas=(betas[0],.999))
        #optim_g = torch.optim.AdamW(G.parameters(),lrs[1],betas=(betas[1],.999))
    
        optim_d = torch.optim.SGD(D.parameters(),lrs[0],momentum=momentum,weight_decay=lrs[0]/10)
        optim_g = torch.optim.SGD(G.parameters(),lrs[1],momentum=momentum,weight_decay=lrs[1]/10)
        
        

        
        t.c_exec(load,ep,bs,optim_d,optim_g,files,ncz,outsize,series_path,5,verbose=True,sample_rate=1)
