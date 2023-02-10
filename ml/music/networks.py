from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from finder import total_size_c

class FullyConnectedNetwork(nn.Module):
    def __init__(self,input_size,output_size,loss_fn=None,optimizer_fn=None,lr=1e-6,wd=1e-6,architecture=[512,32,16]):
        super(FullyConnectedNetwork,self).__init__()

        self.model = nn.Sequential(nn.Linear(input_size,architecture[0]))
        self.model.append(nn.LeakyReLU(.5))

        for i,size in enumerate(architecture[:-1]):

            self.model.append(nn.Linear(size,architecture[i+1]))
            self.model.append(nn.LeakyReLU(.5))
        self.model.append(nn.Linear(architecture[-1],output_size))
        self.model.append(nn.Softmax(dim=0))
        self.optimizer = optimizer_fn(self.model.parameters(),lr=lr,weight_decay=wd)
        self.loss = loss_fn()

    def train(self,x_input,y_actual,epochs=1000,verbose=False,show_steps=10,batch_size="online",show_graph=False):
        memory = 3
        prev_loss = [100000000 for x in range(memory)]
        losses = []
        if type(batch_size) is str:
            batch_size = len(y_actual)

        if verbose:
            print(f"Training on dataset shape:\t f{x_input.shape} -> {y_actual.shape}")
            print(f"batching size:\t{batch_size}")

        #Create the learning batches
        dataset = torch.utils.data.TensorDataset(x_input,y_actual)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)


        for i in range(epochs):
            #Track loss avg in batch
            avg_loss = 0

            for batch_i, (x,y) in enumerate(dataloader):

                #Find the predicted values
                batch_prediction = self.forward(x)
                #Calculate loss
                loss = self.loss(batch_prediction,y)
                avg_loss += loss
                #Perform grad descent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = avg_loss / batch_i 			# Add losses to metric's list
            losses.append(avg_loss.cpu().detach().numpy())

            #Check for rising error
            if not False in [prev_loss[x] > prev_loss[x+1] for x in range(len(prev_loss)-1)]:
                print(f"broke on epoch {i}")
                break
            else:
                prev_loss = [avg_loss] + [prev_loss[x+1] for x in range(len(prev_loss)-1)]

            #Check for verbosity
            if verbose and i % show_steps == 0:
                print(f"loss on epoch {i}:\t{loss}")

        if show_graph:
            plt.plot(losses)
            plt.show()


    def forward(self,x_list):
        return self.model(x_list)
        #	y_predicted.append(y_pred.cpu().detach().numpy())

class ConvolutionalNetwork(nn.Module):
    
    def __init__(self,loss_fn=None,optimizer_fn=None,lr=1e-6,wd:float=1e-6,architecture:list=[[3,2,5,3,2]],input_shape=(1,3,30,20)):
        super(ConvolutionalNetwork,self).__init__()
        self.model = []
        switched = False 
        self.input_shape = input_shape

        self.activation = {	"relu" : nn.ReLU,
                            "sigmoid" : nn.Sigmoid}

        for i,layer in enumerate(architecture):
            if len(layer) == 3:
                in_c,out_c,kernel = layer[0],layer[1],layer[2]
                self.model.append(nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=kernel,padding=2))
                self.model.append(self.activation['relu']())
                
            else:
                in_size, out_size = layer[0],layer[1]
                if not switched:
                    self.model.append(nn.Flatten(1))
                    switched = True 
                self.model.append(nn.Linear(in_size,out_size))
                if not i == len(architecture)-1 :
                    self.model.append(self.activation['relu']())

        #self.model.append(nn.Softmax(1))
        o_d = OrderedDict({str(i) : n for i,n in enumerate(self.model)})
        self.model = nn.Sequential(o_d)
        self.loss = loss_fn()
        self.optimizer = optimizer_fn(self.model.parameters(),lr=lr)
        
    def train(self,x_input,y_actual,epochs=10,in_shape=(1,6,10,10)):

        #Run epochs
        for i in range(epochs):

            #Predict on x : M(x) -> y
            y_pred = self.model(x_input)
            #Find loss  = y_actual - y
            loss = self.loss_function(y_pred,y_actual)
            print(f"epoch {i}:\nloss = {loss}")

            #Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self,x):
        #if len(x.shape) == 3:
            #input(f"received input shape: {x.shape}")
            #x = torch.reshape(x,self.input_shape)
        
        return self.model(x)


class ConvNet(nn.Module):
    def __init__(self,architecture,loss_fn=nn.MSELoss,optimizer=torch.optim.SGD,lr=.005):
        self.layers = {}

        for l in architecture:
            #Add Conv2d layers
            if len(l) == 3:
                in_c,out_c,kernel_size = l[0],l[1],l[2]
                self.layers.append(len(self.layers),nn.Conv2d(in_c,out_c,kernel_size))
                self.layers.append(nn.ReLU())
            #Add Linear layers
            elif len(l) == 2:
                in_dim,out_dim = l[0],l[1]
                self.layers[len(self.layers) : nn.Linear(in_dim,out_dim)]
                if not l == architecture[-1]:
                    self.layers.append(nn.ReLU())
        
        self.loss = loss_fn()


class AudioGenerator(nn.Module):

    #	2 MIN SONGS, BITRATE IS 44100, must be 										
    def __init__(self,num_channels=2,kernels=[16,11,9,3],channels=[8,4,2],strides=[16,15,1,1],paddings=[0,1,1,1],out_pads=[],device=torch.device("cpu")):
        super(AudioGenerator, self).__init__()

        model = OrderedDict()

        for i,config in enumerate(zip(channels,kernels,strides,paddings)):
            f,k,s,p = config 
            model[str(3*i)]         = nn.ConvTranspose1d(channels[i],channels[i+1],kernels[i],strides[i],paddings[i],output_padding=out_pads[i],bias=False)
            model[str(3*i+1)]       = nn.BatchNorm1d(channels[i+1])
            model[str(3*i+2)]       = nn.ReLU(True)

        model[str(3*i+3)] = nn.Tanh()
        self.model = nn.Sequential(model)    

        self.to(device)
        self.device = device
        self.num_channels = num_channels

    def forward(self, input):
        out = self.model(input)
        return out

class AudioGenerator2(nn.Module):

    def __init__(self,factors,channels,scales):
        
        #Parent Class 
        super(AudioGenerator2,self).__init__()

        #Build model
        self.model  = nn.Sequential()
        
        for i in range(len(factors)):
            self.model.append(nn.Upsample(scale_factor=factors[i]*scales[i]))
            self.model.append(nn.Conv1d(channels[i],channels[i+1],factors[i],scales[i]))
            self.model.append(nn.ReLU(True))

            if i == len(factors)-1:
                self.model.append(nn.Tanh())
            else:
                self.model.append(nn.LeakyReLU(.2,inplace=False))
        

    
    def forward(self,x):
        return self.model(x)

class AudioDiscriminator(nn.Module):
    def __init__(self,channels=[2,8,4,4,4,2,2,1,1],kernels=[22050,5000,10,5],strides=[1,1,1,1],paddings=[],final_layer=0,device=torch.device("cpu"),verbose=False):
        super(AudioDiscriminator, self).__init__()
        
        if not len(channels) == len(kernels)+1:
            print(f"bad channel size {len(channels)} must be {len(kernels)+1}")
            exit(-1)
        model = OrderedDict()
        for i,config in enumerate(zip(channels,kernels,strides,paddings)):
            f,k,s,p = config 

            model[str(3*i)]         = nn.Conv1d(channels[i],channels[i+1],kernels[i],strides[i],paddings[i],bias=False)
            if not (i == (len(channels)-2)):
                model[str(3*i+1)]       = nn.BatchNorm1d(channels[i+1])
                model[str(3*i+2)]       = nn.LeakyReLU(.5,True)

            else:
                if final_layer > 1:
                    model[str(3*i+3)]   = nn.Flatten()
                    model[str(3*i+4)]   = nn.Linear(final_layer,1)
                    model[str(3*i+5)]   = nn.Sigmoid()
                    
                else:
                    model[str(3*i+3)]       = nn.Flatten()
                    model[str(3*i+4)]       = nn.Sigmoid()
                    

        self.model = nn.Sequential(model)
        if verbose:
            print(self.model)
        self.model.to(device)
        self.channels = channels

    def forward(self, input):
        return self.model(input)


class AudioDiscriminator2(nn.Module):
    def __init__(self,channels=[2,8,4,4,4,2,2,1,1],kernels=[22050,5000,10,5],mp_kernels=[1,1,1,1],paddings=[],final_layer=0,device=torch.device("cpu"),verbose=False):
        super(AudioDiscriminator2, self).__init__()

        if not len(channels) == len(kernels)+1:
            print(f"bad channel size {len(channels)} must be {len(kernels)+1}")
            exit(-1)

        model = OrderedDict()
        for i,config in enumerate(zip(channels,kernels,mp_kernels,paddings)):
            f,k,s,p = config 


            if not (i == (len(channels)-2)):
                model[str(4*i)]         = nn.Conv1d(channels[i],channels[i+1],kernels[i],1,paddings[i],bias=False)
                model[str(4*i+1)]       = nn.BatchNorm1d(channels[i+1])
                model[str(4*i+2)]       = nn.LeakyReLU(.2,False)
                model[str(4*i+3)]       = nn.MaxPool1d(mp_kernels[i+1])

            else:
                if final_layer > 1:
                    model[str(4*i)]         = nn.Conv1d(channels[i],channels[i+1],kernels[i], 1,paddings[i],bias=False)
                    model[str(4*i+1)]       = nn.BatchNorm1d(channels[i+1])
                    model[str(4*i+1)]       = nn.LeakyReLU(.2,False)
                    model[str(4*i+2)]       = nn.Flatten()
                    model[str(4*i+3)]       = nn.Linear(final_layer,1)
                    model[str(4*i+4)]       = nn.Sigmoid()
                    
                else:
                    model[str(4*i)]         = nn.Conv1d(channels[i],channels[i+1],kernels[i],1,paddings[i],bias=True)
                    model[str(4*i+1)]       = nn.Flatten()
                    model[str(4*i+2)]       = nn.Sigmoid()
                    

        self.model = nn.Sequential(model)
        if verbose:
            print(self.model)
        self.model.to(device)
        self.channels = channels

    def forward(self, input):
        return self.model(input)


class AudioDiscriminator3(nn.Module):
    def __init__(self,channels=[2,8,4,4,4,2,2,1,1],kernels=[22050,5000,10,5],mp_kernels=[1,1,1,1],paddings=[],final_layer=0,device=torch.device("cpu"),verbose=False):
        super(AudioDiscriminator3, self).__init__()

        if not len(channels) == len(kernels)+1:
            print(f"bad channel size {len(channels)} must be {len(kernels)+1}")
            exit(-1)

        model = OrderedDict()
        for i,config in enumerate(zip(channels,kernels,mp_kernels,paddings)):
            f,k,s,p = config 


            if not (i == (len(channels)-2)):
                model[str(4*i)]         = nn.Conv1d(channels[i],channels[i+1],kernels[i],1 if i == 0 else 3,paddings[i],bias=False)
                model[str(4*i+1)]       = nn.BatchNorm1d(channels[i+1])
                model[str(4*i+2)]       = nn.LeakyReLU(.2,False)
                model[str(4*i+3)]       = nn.MaxPool1d(mp_kernels[i+1])

            else:
                if final_layer > 1:
                    model[str(4*i)]         = nn.Conv1d(channels[i],channels[i+1],kernels[i],1,paddings[i],bias=True)
                    model[str(4*i+1)]       = nn.LeakyReLU(.2,False)
                    model[str(4*i+2)]       = nn.Flatten()
                    model[str(4*i+3)]       = nn.Linear(final_layer,1)
                    model[str(4*i+4)]       = nn.Sigmoid()
                    
                else:
                    model[str(4*i)]         = nn.Conv1d(channels[i],channels[i+1],kernels[i],1,paddings[i],bias=True)
                    model[str(4*i+1)]       = nn.Flatten()
                    model[str(4*i+2)]       = nn.Sigmoid()
                    

        self.model = nn.Sequential(model)
        if verbose:
            print(self.model)
        self.model.to(device)
        self.channels = channels

    def forward(self, input):
        return self.model(input)

if __name__ == "__main__":
    nc          = 5
    config      = {"kernels":[8192, 8192, 64, 4, 4, 4, 4], "strides": [3, 3, 3, 3, 3, 3, 2], "paddings": [0, 0, 0, 0, 0, 0, 0], "out_pad":[0, 0, 0, 0, 0, 0, 0], "in_size":1, "num_channels":2}
    kernels_ct  = config['kernels']
    strides_ct  = config['strides']
    padding_ct  = config['paddings']
    out_pad_ct  = config['out_pad']

    channels   = [nc] + [2]*len(padding_ct)
    channels[1] = 8 
    channels[2] = 4
    g = AudioGenerator(channels=channels,kernels=kernels_ct,strides=strides_ct,paddings=padding_ct,out_pads=out_pad_ct,device=torch.device('cuda'))
    print(g)
    import torch 

    inputvect   = torch.randn(size=(1,nc,1),dtype=torch.float,device=torch.device('cuda'))
    out         = g.forward(inputvect)[0]

    print(out.shape)
