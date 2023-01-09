import torch 
from torch.nn import ConvTranspose1d,Upsample,BatchNorm1d,ReLU,LeakyReLU,Conv1d,Sequential,Tanh
from networks import AudioGenerator,AudioDiscriminator

from utilities import model_size

from torch.nn import Upsample

base_factors = [2,2,2,2,2,3,3,3,5,5,5,7,7] 
#           IDEA 1 -
# Use a ConvTranspose with strides equal to factors of input    
ncz = 512 
random_input    = torch.randn(size=(1,ncz,1))


G1      = Sequential(  
            ConvTranspose1d(ncz,2,7,7),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,7,7),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,5,5),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,5,5),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,3,3),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,3,3),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,3,3),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,2,2),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,2,2),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,2,2),
            BatchNorm1d(2),
            LeakyReLU(.02,True),
            ConvTranspose1d(2,2,2,2),
            Tanh())

g1c     = [512,128,128,64,64,64,64,64,64,32] 

G1_OPP  = Sequential(  
            ConvTranspose1d(ncz,g1c[0],2,2),
            BatchNorm1d(g1c[0]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[0],g1c[1],2,2),
            BatchNorm1d(g1c[1]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[1],g1c[2],2,2),
            BatchNorm1d(g1c[2]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[2],g1c[3],2,2),
            BatchNorm1d(g1c[3]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[3],g1c[4],3,3),
            BatchNorm1d(g1c[4]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[4],g1c[5],3,3),
            BatchNorm1d(g1c[5]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[5],g1c[6],3,3),
            BatchNorm1d(g1c[6]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[6],g1c[7],5,5),
            BatchNorm1d(g1c[7]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[7],g1c[8],5,5),
            BatchNorm1d(g1c[8]),
            LeakyReLU(.02,True),
            ConvTranspose1d(g1c[8],g1c[9],7,7),
            BatchNorm1d(g1c[9]),
            LeakyReLU(.02,True),        
            ConvTranspose1d(g1c[9],2,7,7),
            BatchNorm1d(2),
            
            Tanh())

if __name__ == "__main__":
    print(f"Outshape is {G1_OPP(random_input).shape}")
    print(model_size(G1))