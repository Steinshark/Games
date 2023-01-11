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


def build_gen(ncz=512,leak=.02,reverse_factors=False,reverse_channels=False,device=torch.device('cuda'),ver=1):
    factors     = [2,2,2,2,3,3,3,5,5,7,7]
    channels    = [8192,4096,1024,1024,1024,1024,1024,512,512,512] 

    #CH HI to LOW 
    if reverse_factors and not reverse_channels:
        factors.reverse()
        channels    = [4096,2048,2048,1024,512,512,128,128,128,128] 
    #CH LOW to HI 
    elif reverse_factors and reverse_channels:
        factors.reverse()
        channels    = [32,32,32,64,64,64,128,128,128,128]
    #CH HI TO LOW 
    
    elif not reverse_factors and not reverse_channels:
        if ver == 1:
            channels = [4096,4096,2048,1024,512,256,128,64,16,4] 
        if ver == 2:
            channels = [1024*8,1024*4,1024*4,1024*2,1024,1024,512,256,256,128]
        elif ver == 3:
            channels = [1024*16,1024*8,1024*4,1024,512,512,512,256,256,256] 
    
    #CH LOW TO HI 
    elif not reverse_factors and reverse_channels:
        if ver == 1:
            channels = [8,16,32,64,128,128,128,128,128,128] 
        if ver == 2:
            channels = [64,64,64,64,128,128,128,256,256,512]
        elif ver == 3:
            channels = [32,32,32,32,32,64,64,128,128,256] 
    
    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],factors[0],factors[0]),
                            BatchNorm1d(channels[0]),
                            LeakyReLU(leak,True))

    for i,ch in enumerate(channels):
        if i+1 == len(channels):
            Gen.append(         ConvTranspose1d(ch,2,factors[i+1],factors[i+1]))
            Gen.append(         Tanh())

        else:
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1])) 
            Gen.append(         LeakyReLU(leak,True))   
    return Gen.to(device)


def build_short_gen(ncz=512,leak=.02,reverse_factors=False,reverse_channels=False,device=torch.device('cuda'),ver=1):
    factors     = [2,5,7,7,8,9,15]

    #CH HI to LOW 
    if reverse_factors and reverse_channels:
        factors.reverse()
        channels    = [16,32,64,128,512,1204] 
    
    #CH LOW to HI 
    elif reverse_factors and not reverse_channels:
        factors.reverse()
        channels    = [2048,1024,512,256,128,64]
    
    #CH HI TO LOW 
    elif not reverse_factors and reverse_channels:
        channels    = [16,32,64,128,512,1204]
    
    #CH LOW TO HI 
    elif not reverse_factors and not reverse_channels:
        channels    = [2048,1024,512,256,128,64]
    
    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],factors[0],factors[0]),
                            BatchNorm1d(channels[0]),
                            LeakyReLU(leak,True))
    
    for i,ch in enumerate(channels):
        if i+1 == len(channels):
            Gen.append(         ConvTranspose1d(ch,2,factors[i+1],factors[i+1]))
            Gen.append(         Tanh())

        else:
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1])) 
            Gen.append(         LeakyReLU(leak,True))  
    
    return Gen.to(device)


def build_gen2(ncz=512,leak=.02,kernel=33,pad=16,device=torch.device('cuda')):
    factors     = [2,2,2,2,3,3,3,5,5,7,7]
    #channels    = [4096,4096,1024,1024,512,512,1282,128,128,64,64] 
    channels    = [8192,4096,1024,1024,512,512,128,128,128,64,64] 
    
    Gen     = Sequential(   Upsample(ncz,channels[0],factors[0],factors[0]),
                           # Conv1d()
                            BatchNorm1d(channels[0]),
                            LeakyReLU(.02,True))

    for i,ch in enumerate(channels):
        if not i+2 == len(channels):
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         LeakyReLU(leak,True))
             
        else:
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         LeakyReLU(leak,True))
            Gen.append(         Conv1d(channels[i+1],2,kernel_size=kernel,stride=1,padding=pad))
            Gen.append(         Tanh())
            break

    
    return Gen.to(device)


def build_short_gen2(ncz=512,leak=.02,reverse_factors=False,reverse_channels=False,kernel=33,pad=16,device=torch.device('cuda'),ver=1):
    factors     = [2,5,7,7,8,9,15]

    #CH HI to LOW 
    if reverse_factors and reverse_channels:
        factors.reverse()
        channels    = [16,32,64,128,512,1204,1204] 
    
    #CH LOW to HI 
    elif reverse_factors and not reverse_channels:
        factors.reverse()
        channels    = [2048,1024,512,256,128,64,64]
    
    #CH HI TO LOW 
    elif not reverse_factors and reverse_channels:
        channels    = [16,32,64,128,512,1024,1024]
    
    #CH LOW TO HI 
    elif not reverse_factors and not reverse_channels:
        channels    = [2048,1024,512,256,128,64,64]
    
    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],factors[0],factors[0]),
                            BatchNorm1d(channels[0]),
                            LeakyReLU(leak,True))
    
    for i,ch in enumerate(channels):
        if i+2 == len(channels):
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1])) 
            Gen.append(         LeakyReLU(leak,True)) 
            Gen.append(         Conv1d(channels[i+1],channels[i+2],kernel_size=kernel,stride=1,padding=pad))
            Gen.append(         Tanh())

        else:
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1])) 
            Gen.append(         LeakyReLU(leak,True))  
    
    return Gen.to(device)



#EXTRACT AUDIO FEATURES FIRST, THEN GO BACK UP 
def build_encdec(ncz,bs):
    factors         = [2,2,2,2,3,3,3,5,5,7,7]
    output          = 529_200

    #Start with 2 encoder layers
    enc_kernels     = [9,9,17]
    enc_channels    = [128,256]
    enc_strides     = [4,2] 
    G   = Sequential(   Conv1d(ncz,             enc_channels[0],enc_kernels[0],stride=enc_strides[0],padding=0),
                        BatchNorm1d(enc_channels[0]),
                        LeakyReLU(.2,True),

                        Conv1d(enc_channels[0], enc_channels[1],enc_kernels[2],stride=enc_strides[1],padding=0),
                        BatchNorm1d(enc_channels[2]),
                        LeakyReLU(.2,True),
                        )


    #Finish with decoder layers 
    decoding_factors= [2,3,3,3,5,5,7,7]
    dec_channels    = [256,256,256,128,64,32]
    G.append(           Upsample(size=[bs,]))



if __name__ == "__main__":
    g = build_short_gen(ncz=512)
    print(g)
    print(model_size(g))

    rands = torch.randn(size=(1,512,1),device=torch.device("cuda"))
    print(g(rands).shape)