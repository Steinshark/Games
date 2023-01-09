import torch 
from torch.nn import ConvTranspose1d,Upsample,BatchNorm1d,ReLU,LeakyReLU,Conv1d,Sequential,Tanh
from networks import AudioGenerator,AudioDiscriminator

from utilities import model_size

from torch.nn import Upsample

#base_factors = [2,2,2,2,2,3,3,3,5,5,5,7,7] 
#           IDEA 1 -
# Use a ConvTranspose with strides equal to factors of input    
#ncz = 512 
#random_input    = torch.randn(size=(1,ncz,1))


def build_gen(ncz=512,leak=.02,reverse_factors=False,reverse_channels=False,device=torch.device('cuda')):
    factors     = [2,2,2,2,3,3,3,5,5,7,7]
    channels    = [8192,4096,1024,1024,1024,1024,1024,512,512,512] 

    if reverse_factors and reverse_channels:
        factors.reverse()
        channels    = [4096,2048,2048,1024,512,512,128,128,128,128] 
    elif reverse_factors and not reverse_channels:
        factors.reverse()
        channels    = [32,32,32,64,64,64,128,128,128,128]
    elif not reverse_factors and reverse_channels:
        channels = [8192,4096,1024,1024,1024,1024,1024,512,512,512] 
    elif not reverse_factors and not reverse_channels:
        channels    = [8192,4096,1024,1024,1024,1024,1024,512,512,512]
    
    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],factors[0],factors[0]),
                            BatchNorm1d(channels[0]),
                            LeakyReLU(.02,True))
    for i,ch in enumerate(channels):
        if i+1 == len(channels):
            next_ch     = 2
            add_relu    = False 
        else:
            next_ch     = channels[i+1] 
            add_relu    = True 
        Gen.append(         ConvTranspose1d(ch,next_ch,factors[i+1],factors[i+1]))
        Gen.append(         BatchNorm1d(next_ch))
        
        if add_relu:
            Gen.append(         LeakyReLU(leak,True))
        else:
            Gen.append(         Tanh())
    
    return Gen.to(device)

if __name__ == "__main__":
    print(build_gen(reverse_channels=False,reverse_factors=True))