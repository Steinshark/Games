import torch 
from torch.utils.data import Dataset, DataLoader
from networks import AudioGenerator, AudioDiscriminator
import numpy 
import time 
import json
import os 
from dataset import reconstruct
import random 
import sys 
import pprint 
from utilities import weights_init, config_explorer, lookup, G_CONFIGS, D_CONFIGS

class AudioDataSet(Dataset):

    def __init__(self,fnames):

        #Load files as torch tensors 
        self.data = []
        for file in fnames:
            arr = numpy.load(file,allow_pickle=True)
            arr = torch.from_numpy(arr).type(torch.float)
            if not arr.size()[-1] == 5291999:
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

    def __init__(self,device:torch.DeviceObjType,input_channels:int):

        #Create models
        self.Generator      = None 
        self.Discriminator  = None 

        self.device         = device
        self.input_channels = input_channels
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
    def create_models(self,D_config:dict,G_config:dict):



        #Ensure Proper Config Files 
        if not "kernels" in G_config:
            print("Invalid Generator config, must contain Kernel settings")
            return 
        if not "strides" in G_config:
            print("Invalid Generator config, must contain Stride settings")
            return 
        if not "paddings" in G_config:
            print("Invalid Generator config, must contain Padding settings")
            return 
        if not "kernels" in D_config:
            print("Invalid Discrimintator config, must contain Kernel settings")
            return 
        if not "strides" in D_config:
            print("Invalid Discrimintator config, must contain Stride settings")
            return 
        if not "paddings" in D_config:
            print("Invalid Discrimintator config, must contain Padding settings")
            return 
        

        #Create Generator 
        self.Generator   = AudioGenerator(          num_channels=G_config['num_channels'],
                                                    kernels=G_config['kernels'],
                                                    channels=G_config['channels'],
                                                    strides=G_config['strides'],
                                                    paddings=G_config['paddings'],
                                                    out_pads=G_config['out_pad'],
                                                    device=torch.device(G_config['device']))

        #Create Discriminator
        self.Discriminator   = AudioDiscriminator(  channels=D_config['channels'],
                                                    kernels=D_config['kernels'],
                                                    strides=D_config['strides'],
                                                    paddings=D_config['paddings'],
                                                    device=torch.device(G_config['device']),
                                                    verbose=False)

        #Init weights 
        self.Generator.apply(weights_init)
        self.Discriminator.apply(weights_init)

        
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
        self.dataset        = AudioDataSet(filenames)
        self.dataloader     = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

    def set_learners(self,D_optim,G_optim,error_fn):
        self.D_optim    = D_optim
        self.G_optim    = G_optim
        self.error_fn   = error_fn
   
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
            random_inputs           = torch.randn(size=(x_len,self.input_channels,1),dtype=torch.float,device=self.device)
            generator_outputs       = self.Generator.forward(random_inputs)
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

            self.Generator.zero_grad()

            #Classify the fakes again after Discriminator got updated 
            t_0 = time.time()
            fake_class2             = self.Discriminator.forward(generator_outputs).view(-1)
            d_fake2                 += fake_class2.mean().item()
            t_d[-1] += time.time()-t_0
            
            #Classify random vector 
            random_vect             = torch.randn(size=(1,2,5291999),dtype=torch.float,device=self.device)
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

    def print_epoch_header(epoch_num,epoch_tot,header_width=100):
        print("="*header_width)
        epoch_seg   = f'=   EPOCH{f"{epoch_num+1}/{epoch_tot}".rjust(8)}'
        print(epoch_seg,end="")
        print(" "*(header_width-(len(epoch_seg)+1)),end="=\n")
        print("="*header_width)

    def sample(self,out_file_path):
        inputs = torch.randn(size=(1,self.input_channels,1),dtype=torch.float,device=self.device)
        outputs = self.Generator.forward(inputs)
        outputs = outputs[0].cpu().detach().numpy()
        reconstruct(outputs,out_file_path)
        print(f"saved audio to {out_file_path}")



if __name__ == "__main__" and True:

    if len(sys.argv) > 1:
        load        = int(sys.argv[1])
        epochs      = int(sys.argv[2])
        bs          = int(sys.argv[3])
        if len(sys.argv) > 4:
            accu_bs     = int(sys.argv[4])

    else:
        load        = 256
        epochs      = 4 
        bs          = 8

    input_channels          = 100
    configs                 = json.loads(open("configs3.txt","r").read())
    d_configs               = [con for i,con in enumerate(configs['D'])]
    g_configs               = [con for i,con in enumerate(configs['G'])]

    #Create D config 
    d_config                = D_CONFIGS['LoHi_5']
    d_config['channels']    = [2] + [16] + [16] + [4]*(len(d_config['kernels'])-4) + [4] + [1]
    d_config['device']      = 'cuda'    

    #Create G config 
    g_config                = G_CONFIGS['HiLo_2']
    g_config['channels']    = [input_channels] + [128] + [128] + [32]*(len(g_config['kernels'])-4) + [32] + [2]
    g_config['device']      = 'cuda'

    #Build trainer obj
    trainer = Trainer(torch.device('cuda'),input_channels)
    trainer.create_models(d_config,g_config)
    #trainer.load_models(d_config,g_config,"models/D_batch_LoHi_5","models/G_batch_HiLo_1")
    print(f"Discriminator init with size {(sum([ p.numel()*p.element_size() for p in trainer.Discriminator.parameters()])/(1024*1024)):.2f}MB")
    print(f"Generator     init with size {(sum([ p.numel()*p.element_size() for p in trainer.Generator.parameters()])/(1024*1024)):.2f}MB")

    #Place learning items
    d_params    = trainer.Discriminator.parameters()
    g_params    = trainer.Generator.parameters()
    trainer.set_learners(torch.optim.Adam(d_params,lr=.0002,betas=(.75,.999)),torch.optim.Adam(g_params,lr=.0003,betas=(.75,.999)),torch.nn.BCELoss())

    #Train!
    print(f"Starting model training")
    time.sleep(2)
    for ep in range(epochs):
        #Generate training data 
        root        = "C:/data/music/dataset"
        filenames   = os.listdir(root)
        filenames   = random.sample(filenames,load)
        trainer.build_dataset([os.path.join(root,fname) for fname in filenames],bs,True,2)
        Trainer.print_epoch_header(ep,epochs)
        trainer.train()
    
    trainer.save_model_states("models",D_name="D_batch_LoHi_5",G_name="G_batch_HiLo_2")

    trainer.sample(os.path.join("outputs",f"{input('save sample as: ')}.wav"))

