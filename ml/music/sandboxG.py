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


def build_gen(ncz=512,leak=.02,kernel_ver=0,fact_ver=0,device=torch.device('cuda'),ver=1):

    factors     = [[2,2,2,2,3,3,3,5,5,7,7], [7,7,5,5,3,3,3,2,2,2,2],[7,2,2,7,2,2,5,3,5,3,3]][fact_ver]
    channels    = [1024,512,256,128,64,32,32,32,32,32,2] 

    kernels = [
            [65,    65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65],                 #   MED 
            [3,     5,  5,  9,  13, 13, 17, 17, 25, 25, 33, 35, 37, 33],                #   LG-SM 
            [101,   201,251,301,251,201,151,101,51, 41, 31, 21, 11, 5],         #   MED-LG-SM
            [19,    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19],                #   MED_SM
            [3,     7,  9,  11, 15, 19, 21, 25, 35, 45, 55, 65, 75, 85]
    ][kernel_ver]
   
    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],factors[0],factors[0]),
                            BatchNorm1d(channels[0]),
                            LeakyReLU(leak,True))

    Gen.append(         Conv1d(channels[0],channels[0],3,1,1))
    Gen.append(         BatchNorm1d(channels[0]))
    Gen.append(         LeakyReLU(leak,True)) 

    for i,ch in enumerate(channels):
        if i+2 == len(channels):
            Gen.append(         ConvTranspose1d(ch,2,factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(    channels[i+1]))
            Gen.append(         LeakyReLU(leak,True)) 
            Gen.append(         Conv1d(         channels[i+1],channels[i+1],kernels[i],1,padding=int(kernels[i]/2),bias=False))
            Gen.append(         Tanh())
            break

        else:
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1])) 
            Gen.append(         LeakyReLU(leak,True))   

            Gen.append(         Conv1d(         channels[i+1],channels[i+1],kernels[i],1,padding=int(kernels[i]/2),bias=False))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         LeakyReLU(leak,True))
    return Gen.to(device)


def build_short_gen(ncz=512,leak=.2,kernel_ver=0,fact_ver=1,device=torch.device('cuda'),out_ch=2):
    factors     = [15,2,5,8,9,49]

    ch          = [2048,2048,2048,2048,1024]

    ker         = [3,7,15,65,101,501]


    final_ch1    = 444
    final_ch2    = 64 
    final_kern1  = 7
    #final_kern2  = 11

    pad         = [int(k/2) for k in ker] 
    Gen     = Sequential(   ConvTranspose1d(ncz,ch[0],factors[0],factors[0]),
                            BatchNorm1d(ch[0],momentum=.5),
                            LeakyReLU(leak,True))

    #Gen.append(         Conv1d(ch[0],ch[0],3,1,1))
    #Gen.append(         BatchNorm1d(ch[0]))
    #Gen.append(         LeakyReLU(leak,True)) 
    
    for i,c in enumerate(ch):
        if i+1 == len(ch):
            Gen.append(         ConvTranspose1d(c,final_ch1,factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(final_ch1,momentum=.5))
            Gen.append(         LeakyReLU(leak,True))

            #Gen.append(         Conv1d(final_ch1,final_ch2,final_kern1,1,int(final_kern1/2)))
            #Gen.append(         BatchNorm1d(final_ch2,momentum=.5))
            #Gen.append(         LeakyReLU(leak,True))

            Gen.append(         Conv1d(final_ch1,out_ch,final_kern1,1,int(final_kern1/2)))
            Gen.append(         Tanh())

        else:
            Gen.append(         ConvTranspose1d(c,ch[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(ch[i+1],momentum=.5)) 
            Gen.append(         LeakyReLU(leak,True)) 

            #Gen.append(         Conv1d(ch[i+1],ch[i+1],ker[i],1,pad[i],bias=False))
            #Gen.append(         BatchNorm1d(ch[i+1]))
            #Gen.append(         LeakyReLU(leak,True)) 
    
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


def build_encdec(ncz,encoder_factors=[2,3],encoder_kernels=[5,7],dec_factors=[7,5,5,3,3],enc_channels=[256,1024],dec_kernels=[5,17,25,89,513],leak=.2,batchnorm=True):
    
    #Factors management 
    factors             = [2,2,2,2,3,3,3,5,5,7,7]
    factors             = [2,3,8,9,25,49]
    output              = 529_200

    #Start with 2 encoder layers
    enc_channels        = [64,512]

    #Start Generator Architecture with the encoder 
    G   = Sequential(   Conv1d(1,             enc_channels[0],    encoder_kernels[0], stride=encoder_factors[0],padding=int(encoder_kernels[0]/2)))
    if batchnorm:
        G.append(           BatchNorm1d(enc_channels[0]))
    G.append(           LeakyReLU(leak,True))

    G.append(           Conv1d(enc_channels[0], enc_channels[1],    encoder_kernels[1], stride=encoder_factors[1],padding=int(encoder_kernels[1]/2)))
    if batchnorm:
        G.append(           BatchNorm1d(enc_channels[1]))
    G.append(           LeakyReLU(leak,True))
                        

    #Finish with decoder layers 
    dec_channels        = [1024,512,128,64,2]             #OLD 
    dec_conv_kernels    = dec_kernels
    dec_conv_padding    = [int(ker/2) for ker in dec_conv_kernels]

    for i,fact in enumerate(dec_factors):

        #Add conv transpose layer 
        if i == 0:
            G.append(           ConvTranspose1d(enc_channels[-1],dec_channels[i],dec_factors[i],dec_factors[i],padding=0))
        else:
            G.append(           ConvTranspose1d(dec_channels[i-1],dec_channels[i],dec_factors[i],dec_factors[i],padding=0))
      
        
        if i == len(dec_factors)-1:
            G.append(Tanh())
        else:
            #Add rest of layers 
            if batchnorm:
                G.append(           BatchNorm1d(dec_channels[i]))
            G.append(           LeakyReLU(leak,True))

            G.append(           Conv1d(dec_channels[i],dec_channels[i],dec_conv_kernels[i],1,dec_conv_padding[i],bias=False))
            if batchnorm:
                G.append(           BatchNorm1d(dec_channels[i]))
            G.append(           LeakyReLU(leak,True))

    return G.to(torch.device("cuda"))



if __name__ == "__main__":
    kernels     = [15,17,21,23,25,7,5]
    paddings    = [int(k/2) for k in kernels]
    D2      = AudioDiscriminator(channels=[2,32,64,128,256,512,1024,1],kernels=kernels,strides=[12,9,7,7,5,5,4],paddings=paddings,device=torch.device('cuda'),final_layer=1,verbose=False)
    inp = torch.randn((1,2,529200),device=torch.device('cuda'))

    print(f"shape: {D2.forward(inp).item()}")