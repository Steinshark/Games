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

MODELS_PATH      = "D:/code/projects/ml/music/models"
DATASET_PATH    = r"S:\Data\music\dataset"

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
    config = json.loads(open(config_file,"r").read())

    #Ensure file path exists 
    if not os.path.isdir("models"):
        os.mkdir("models")

    #build and load weights for model
    exported_model      = AudioGenerator(   config['in_size'],  
                                            num_channels=config['num_channels'],
                                            kernel_sizes=config['kernel_sizes'],
                                            scales=config['scales'],
                                            strides=config['strides'],
                                            device=config['device'])
    full_path       = os.path.join(MODELS_PATH,fname)
    exported_model.load_state_dict(torch.load(full_path))

    return exported_model


def save_generator(fname,config,model:torch.nn.Module):

    #Save settings 
    with open(f"{fname}_config","w") as config_file:
        config_file.write(json.dumps(config))
        config_file.close()

    #Save model 
    torch.save(model.state_dict(),os.path.join(MODELS_PATH,fname))
    return 


def import_discriminator(fname,config,config_file):
    #Load config file 
    config = json.loads(open(config_file,"r").read())

    #Ensure file path exists 
    if not os.path.isdir("models"):
        os.mkdir("models")

    #build and load weights for model
    exported_model      = AudioDiscriminator(   filters=config['filter_sizes'],  
                                                kernel_sizes=config['kernel_sizes'])

    full_path       = os.path.join(MODELS_PATH,fname)
    exported_model.load_state_dict(torch.load(full_path))

    return exported_model
 

def save_discriminator(fname,config,model:torch.nn.Module):
    #Save settings 
    with open(f"{fname}_config","w") as config_file:
        config_file.write(json.dumps(config))
        config_file.close()

    #Save model 
    torch.save(model.state_dict() ,os.path.join(MODELS_PATH,fname))
    return 


def train(filepaths,epochs=1,lr=.002,betas={'g':(.5,.999),'d':(.5,.999)},bs=4,verbose=True):
    
    #Create dataset and loader for batching more efficiently
    print("building dataset")
    t0 = time.time()
    dataset     = AudioDataSet(filepaths)
    dataloader  = DataLoader(dataset,batch_size=bs,shuffle=True,num_workers=4)
    print       (f"\tcreated dataset of size {dataset.__len__()} in {(time.time()-t0):.2f}s")
    #Create and prep models
    dev         = torch.device("cuda")
    configs     = json.loads(open("configs","r").read())
    g_config    = configs['g'][0]
    d_config    = configs['d'][0]

    g           = AudioGenerator(in_size=44100,num_channels=2,kernel_sizes=g_config['kernels'],strides=g_config['strides'],padding=g_config['padding'],device=dev) 
    print       (f"initialized Generator with {sum([p.numel() for p in g.parameters()])}")

    d           = AudioDiscriminator(kernels=d_config['kernels'],strides=d_config['strides'],paddings=d_config['padding'],device=dev)
    print       (f"initialized Discriminator with {sum([p.numel() for p in d.model.parameters()])}")

    g_opt       = torch.optim.Adam(g.parameters(),lr=lr['g'],betas=betas['g'])
    d_opt       = torch.optim.Adam(d.parameters(),lr=lr['d'],betas=betas['d'])
    error_fn    = torch.nn.BCELoss()
    
    
    #Ensure models are proper sizes
    d_test      = torch.randn(size=(1,2,5292000),device=dev)
    g_test      = torch.randn(size=(1,1,44100),device=dev)
    
    if not g.forward(g_test).shape == torch.Size([1,2,5292000]):
        print   (f"Generator configured incorrectly\n\toutput size was {g.forward(g_test).shape}, must be 5292000")
    if not d.forward(d_test).shape == torch.Size([1,1]):
        print   (f"Discriminator configured incorrectly\n\toutput size was {d.forward(d_test).shape}, must be 1")


    #RUN EPOCHS
    for e in range(epochs):
        
        #Telemetry
        num_equals 	= 50 
        printed 	= 0
        n_batches   = len(dataloader)
        t_d         = [0] 
        t_g         = [0]
        if verbose:
            header = f"\tEPOCH {e}\t|\tPROGRESS\t[{''.join([str('-') for x in range(num_equals)])}]"
            print(header,end='\n',flush=True)
            print(f'{["=" for c in header]}')

            print(f"\t     \t\t{n_batches} batches  ->\t[",end='',flush=True)

        #RUN BATCHES
        for i, data in enumerate(dataloader,0):

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
            y_set               = torch.ones(size=(bs,),dtype=torch.float,device=dev)
            
            #Back propogate
            t_0 = time.time()
            real_class          = d.forward(x_set).view(-1)
            t_d[-1] += time.time()-t_0

            d_error_real        = error_fn(real_class,y_set)
            d_error_real.backward()
            d_performance_real  = real_class.mean().item()  

            #####################################################################
            #                           TRAIN FAKE                              #
            #####################################################################
            
            #Ask generator to make some samples
            random_inputs           = torch.randn(size=(bs,1,44100),dtype=torch.float,device=dev)
            
            t_0 = time.time()
            generator_outputs       = g.forward(random_inputs)
            t_g[-1] += time.time()-t_0
            fake_labels             = torch.zeros(size=(bs,),dtype=torch.float,device=dev)

            #Ask Discriminator to classify fake samples 
            t_0 = time.time()
            fake_class              = d.forward(generator_outputs).view(-1)
            t_d[-1] += time.time()-t_0
            d_error_fake            = error_fn(fake_class,fake_labels)
            d_error_fake.backward()
            d_performance_fake      = fake_class.mean().item()
            d_performance_cum       = d_error_real+d_error_fake
            
            #Back Prop
            d_opt.step()           

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
            g_error.backward()
            d_performance_fake_2    = fake_class.mean().item()
            
            #Back Prop
            g_opt.step()


        print(f"]\tt_d={t_d[-1]}\tt_g={t_g[-1]}",flush=True)
        print("\n")
        print(f'{["=" for c in header]}')
        t_d.append(0)
        t_g.append(0)


if __name__ == "__main__":
    filepaths = random.sample([os.path.join(DATASET_PATH,f) for f in os.listdir(DATASET_PATH)],10)
    train(filepaths,epochs=1,lr={'g':.0002,'d':.0002},betas={'g':(.5,.999),'d':(.5,.999)},bs=4)