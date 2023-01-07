from networks import AudioGenerator, AudioDiscriminator
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy 
import time
import os 
import json 
import random 
import pprint 
from dataset import reconstruct
import sys 
MODELS_PATH      = r"C:\gitrepos\projects\ml\music\models"
DATASET_PATH    = r"D:\data\music\dataset"
#DATASET_PATH    = r"S:\Data\music\dataset"

class AudioDataSet(Dataset):

    def __init__(self,fnames):

        #Load files as torch tensors 
        self.data = []
        for file in fnames:
            arr = numpy.load(file)
            arr = torch.from_numpy(arr).type(torch.float)
            self.data.append([arr,1])

        print(f"loaded {self.__len__()} samples")
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        x = self.data[i][0]
        y = self.data[i][1]
        return x,y


def import_generator(fname,config_file):
    #Load config file 
    config = json.loads(open(os.path.join(MODELS_PATH,config_file),"r").read())

    #Ensure file path exists 
    if not os.path.isdir("models"):
        os.mkdir("models")

    #build and load weights for model
    exported_model      = AudioGenerator(   config['input_size'],  
                                            num_channels=config['num_channels'],
                                            kernels=config['kernels'],
                                            strides=config['strides'],
                                            paddings=config['padding'],
                                            device=torch.device(config['device']))
    full_path       = os.path.join(MODELS_PATH,fname)
    exported_model.load_state_dict(torch.load(full_path))

    return exported_model


def save_generator(fname,config,model:torch.nn.Module):
    
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    #Save settings 
    with open(os.path.join(MODELS_PATH,f"{fname}_config"),"w") as config_file:
        config_file.write(json.dumps(config))
        config_file.close()

    #Save model 
    torch.save(model.state_dict(),os.path.join(MODELS_PATH,fname))
    return 


def import_discriminator(fname,config_file):
    #Load config file 
    config = json.loads(open(os.path.join(MODELS_PATH,config_file),"r").read())

    #Ensure file path exists 
    if not os.path.isdir("models"):
        os.mkdir("models")

    #build and load weights for model
    exported_model      = AudioDiscriminator(   out_size=config['out_size'],
                                                channels=config['channels'],  
                                                kernels=config['kernels'],
                                                paddings=config['paddings'],
                                                device=config['device'])

    full_path       = os.path.join(MODELS_PATH,fname)
    exported_model.load_state_dict(torch.load(full_path))

    return exported_model
 

def save_discriminator(fname,config,model:torch.nn.Module):
    
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    #Save settings 
    with open(os.path.join(MODELS_PATH,f"{fname}_config"),"w") as config_file:
        config_file.write(json.dumps(config))
        config_file.close()

    #Save model 
    torch.save(model.state_dict() ,os.path.join(MODELS_PATH,fname))
    return 


def sample(g_file,config_file,out_file_path):
    g = import_generator(g_file,config_file)
    print(f"input: {g.input_size}")
    inputs = torch.randn(size=(1,1,g.input_size),dtype=torch.float,device=g.device)
    outputs = g.forward(inputs)
    outputs = outputs[0].cpu().detach().numpy()
    reconstruct(outputs,out_file_path)
    print(f"saved audio to {out_file_path}")


def sample_with_model(g,out_file_path):
    print(f"input: {g.input_size}")
    inputs = torch.randn(size=(1,1,g.input_size),dtype=torch.float,device=g.device)
    outputs = g.forward(inputs)
    outputs = outputs[0].cpu().detach().numpy()
    reconstruct(outputs,out_file_path)
    print(f"saved audio to {out_file_path}")


def train(  filepaths,
            epochs=1,
            lr=.002,
            betas={'g':(.5,.999),'d':(.5,.999)},
            bs=4,
            verbose=True,
            loading=False,
            g_fname="generator_1",
            d_fname="discriminator_1",
            g_conf_fname=None,
            d_conf_fname=None,
            dev=torch.device('cuda')):
    
    #Create dataset and loader for batching more efficiently
    print("building dataset")
    t0 = time.time()
    dataset     = AudioDataSet(filepaths)
    dataloader  = DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=4)
    print       (f"\tcreated dataset of size {dataset.__len__()} in {(time.time()-t0):.2f}s\n\n")
    #Create and prep models
    configs     = json.loads(open("configs.txt","r").read())

    if g_conf_fname == None:
        g_config    = configs['g'][0]
        d_config    = configs['d'][-1]

    if loading:
        g       = import_generator(g_fname,g_conf_fname)
    
    else:
        g           = AudioGenerator(in_size=44100,num_channels=2,kernels=g_config['kernels'],strides=g_config['strides'],paddings=g_config['padding'],device=dev) 
    print       (f"initialized Generator with {sum([p.numel() for p in g.parameters()])}")
    
    if loading:
        d       = import_discriminator(d_fname,d_conf_fname)
    else:
        d           = AudioDiscriminator(out_size=int(d_config['out_size']),kernels=d_config['kernels'],strides=d_config['strides'],paddings=d_config['padding'],device=dev)
    print       (f"initialized Discriminator with {sum([p.numel() for p in d.model.parameters()])}")

    g_opt       = torch.optim.Adam(g.parameters(),lr=lr['g'],betas=betas['g'])
    d_opt       = torch.optim.Adam(d.parameters(),lr=lr['d'],betas=betas['d'])
    error_fn    = torch.nn.BCELoss()
    
    
    #Ensure models are proper sizes
    d_test      = torch.randn(size=(1,2,5292000),dtype=torch.float,device=dev)
    g_test      = torch.randn(size=(1,1,44100),dtype=torch.float,device=dev)
    
    if not g.forward(g_test).shape == torch.Size([1,2,5292000]):
        print   (f"Generator configured incorrectly\n\toutput size was {g.forward(g_test).shape}, must be 5292000")
        exit()
    
    if not d.forward(d_test).shape == torch.Size([1,1]):
        print   (f"Discriminator configured incorrectly\n\toutput size was {d.forward(d_test).shape}, must be 1")
        exit()


    #RUN EPOCHS
    for e in range(epochs):
        
        #Telemetry
        num_equals 	= 50 
        printed 	= 0
        n_batches   = len(dataloader)
        t_d         = [0] 
        t_g         = [0]
        t_op_d      = [0]
        t_op_g      = [0]
        if verbose:
            header = f"\tEPOCH {e}\t|\tPROGRESS\t[{''.join([str(' ') for x in range(num_equals)])}]"
            print("\n\n")
            print("="*(len(header)+20))
            print(header,end='\n',flush=True)
            print("="*(len(header)+20))

            batch_spacer = f"\t     \t\t{n_batches} batches  ->\t["
            print(f"\t     \t\t{n_batches} batches  ->\t[",end='',flush=True)

        #RUN BATCHES
        for i, data in enumerate(dataloader,0):
            t_init = time.time()
            #Telemetry
            if verbose:
                percent = i / n_batches
                while (printed / num_equals) < percent:
                    print("=",end='',flush=True)
                    printed+=1



            #####################################################################
            #                           TRAIN REAL                              #
            #####################################################################
            d.zero_grad()

            #Prep values, all are real valued
            x_set               = data[0].to(dev)
            y_set               = torch.ones(size=(len(x_set),),dtype=torch.float,device=dev)
            
            #Back propogate
            t_0 = time.time()
            real_class          = d.forward(x_set).view(-1)
            t_d[-1] += time.time()-t_0
            d_error_real        = error_fn(real_class,y_set)
            t_0 = time.time()
            d_error_real.backward()
            t_op_d[-1] += time.time()-t_0
            d_performance_real  = real_class.mean().item()  

            #####################################################################
            #                           TRAIN FAKE                              #
            #####################################################################
            
            #Ask generator to make some samples
            random_inputs           = torch.randn(size=(len(x_set),1,44100),dtype=torch.float,device=dev)
            
            t_0 = time.time()
            generator_outputs       = g.forward(random_inputs)
            t_g[-1] += time.time()-t_0
            fake_labels             = torch.zeros(size=(len(x_set),),dtype=torch.float,device=dev)

            #Ask Discriminator to classify fake samples 
            t_0 = time.time()
            fake_class              = d.forward(generator_outputs).view(-1)
            t_d[-1] += time.time()-t_0
            d_error_fake            = error_fn(fake_class,fake_labels)
            t_0 = time.time()
            d_error_fake.backward()
            d_performance_fake      = fake_class.mean().item()
            d_performance_cum       = d_error_real+d_error_fake
            
            #Back Prop
            d_opt.step()           
            t_op_d[-1] += time.time()-t_0

            #####################################################################
            #                           TRAIN GENR                              #
            #####################################################################

            g.zero_grad()
            
            #Classify the fakes again after Discriminator got updated 
            t_0 = time.time()
            fake_class              = d.forward(generator_outputs.detach()).view(-1)
            t_d[-1] += time.time()-t_0
            #Find the error between the fake batch and real set  
            g_error                 = error_fn(fake_class,y_set)
            t_0 = time.time()
            g_error.backward()
            d_performance_fake_2    = fake_class.mean().item()
            
            #Back Prop
            g_opt.step()
            t_op_g[-1] += time.time()-t_0
        
        if verbose:
            percent = (i+1) / n_batches
            while (printed / num_equals) < percent:
                print("=",end='',flush=True)
                printed+=1

        print(f"]\tG forw={sum(t_d):.3f}s\tG forw={sum(t_g):.3f}s\tD back={sum(t_op_d):.3f}s\tG back={sum(t_op_g):.3f}s\ttot = {(time.time()-t_init):.2f}s",flush=True)
        print(" "*(len(batch_spacer)+num_equals),end="")
        print(f"\tD(fake) = {(d_performance_fake):.3f}\tD(fake2) = {(d_performance_fake_2):.3f}")
        print("\n\n")
        t_d.append(0)
        t_g.append(0)

    if dev == torch.device("cuda"):
        g_config['device'] = 'cuda'
        d_config['device'] = 'cuda'
    else:
        g_config['device'] = 'cpu'
        d_config['device'] = 'cpu'
    
    d_config['num_channels']    = d.channels 
    g_config['num_channels']    = g.num_channels
    g_config['in_size']         = 44100
    d_config['out_size']        = int(d_config['out_size'])
    save_generator("generator_1",g_config,g)
    save_discriminator("discriminator_1",d_config,d)

    sample_with_model(g,"Training_day_2_final.wav")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if not len(sys.argv) == 4:
            print("Must be run like: 'python trainer.py [ep] [bs] [dataset_size]'")
            exit()
        
        epochs  = int(sys.argv[1])
        bs      = int(sys.argv[2])
        ds_len  = int(sys.argv[3])
    else:
        epochs  = 2 
        bs      = 4 
        ds_len  = 64

    filepaths = random.sample([os.path.join(DATASET_PATH,f) for f in os.listdir("D:/data/music/dataset")],64)
    train(  filepaths,
            epochs=2,
            lr={'g':.001,'d':.001},
            betas={'g':(.5,.999),'d':(.5,.999)},
            bs=8,
            loading=False,
            g_conf_fname=None,
            d_conf_fname=None,
            dev=torch.device('cuda'))

    sample("generator_1","generator_1_config","training_day_3_first.wav")
#    generator_1_config
#discriminator_1_config