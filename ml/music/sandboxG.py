import torch 
from torch.nn import ConvTranspose1d,Upsample,BatchNorm1d,ReLU,LeakyReLU,Conv1d,Sequential,Tanh, Sigmoid
from networks import AudioGenerator,AudioDiscriminator

from utilities import model_size

from torch.nn import Upsample

#base_factors = [2,2,2,2,2,3,3,3,5,5,5,7,7] 
#           IDEA 1 -
# Use a ConvTranspose with strides equal to factors of input    
#ncz = 512 
#random_input    = torch.randn(size=(1,ncz,1))


def build_gen(ncz=512,leak=.02,kernel_ver=0,fact_ver=0,ch_ver=1,device=torch.device('cuda'),ver=1,out_ch=1):

    factors     = [[2,2,2,2,3,3,3,5,5,7,7], [7,7,5,5,3,3,3,2,2,2,2],[7,2,2,7,2,2,5,3,5,3,3],[4,5,7,7,5,3,3,4]][fact_ver]
    channels    = [[4096,4096,4096,2048,2048,1024,1024,512,256,256,256],[4096,4096,2048,1024,512,1024,512,256]][ch_ver]

    kernels = [
            [5,     5,  9,  21, 21, 19, 17, 17, 17, 11, 11, 11, 11],                #   LG-SM 
            [3,     17, 65, 65 ,65 ,129 ,129],         #   MED-LG-SM
            [19,    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19],                #   MED_SM
    ][kernel_ver]
   
    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],factors[0],factors[0]),
                            BatchNorm1d(channels[0]),
                            LeakyReLU(leak,True))


    for i,ch in enumerate(channels):
        if i+2 == len(channels):
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(    channels[i+1]))
            Gen.append(         LeakyReLU(leak,True)) 

            Gen.append(         Conv1d(         channels[i+1],out_ch,kernels[i],1,padding=int(kernels[i]/2),bias=True))
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

def build_short_gen(ncz=512,leak=.2,momentum=.95,device=torch.device('cuda'),out_ch=2):
    factors     = [49,9,8,5,5,2]

    ch          = [ncz,2000,2000,1000,100]

    ker         = [3,7,15,65,101,501]


    final_ch1    = 100
    final_ch2    = 64 
    final_kern1  = 17
    #final_kern2  = 11

    pad         = [int(k/2) for k in ker] 
    Gen     = Sequential(   ConvTranspose1d(ncz,ch[0],factors[0],factors[0]),
                            BatchNorm1d(ch[0],momentum=momentum),
                            LeakyReLU(leak,True))

    #Gen.append(         Conv1d(ch[0],ch[0],3,1,1))
    #Gen.append(         BatchNorm1d(ch[0]))
    #Gen.append(         LeakyReLU(leak,True)) 
    
    for i,c in enumerate(ch):
        if i+1 == len(ch):
            Gen.append(         ConvTranspose1d(c,final_ch1,factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(final_ch1,momentum=momentum))
            Gen.append(         LeakyReLU(leak,True))

            #Gen.append(         Conv1d(final_ch1,final_ch2,final_kern1,1,int(final_kern1/2)))
            #Gen.append(         BatchNorm1d(final_ch2,momentum=.5))
            #Gen.append(         LeakyReLU(leak,True))

            Gen.append(         Conv1d(final_ch1,out_ch,final_kern1,1,int(final_kern1/2)))
            Gen.append(         Tanh())

        else:
            Gen.append(         ConvTranspose1d(c,ch[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(ch[i+1],momentum=momentum)) 
            Gen.append(         LeakyReLU(leak,True)) 

            #Gen.append(         Conv1d(ch[i+1],ch[i+1],ker[i],1,pad[i],bias=False))
            #Gen.append(         BatchNorm1d(ch[i+1]))
            #Gen.append(         LeakyReLU(leak,True)) 
    
    return Gen.to(device)

def build_sig(ncz=512,out_ch=1,device=torch.device('cuda')):
    factors     = [9,7,7,5,5,4,2,2]
    #channels    = [4096,4096,1024,1024,512,512,1282,128,128,64,64] 
    channels    = [128,256,256,256,256,128,64,32] 
    
    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],factors[0],factors[0]),
                            BatchNorm1d(channels[0]),
                            ReLU())

    kernel_1        = 513 
    kernel_2        = 7
    kernel_3        = 3
    ch_1            = 64
    ch_2            = 20
    for i,ch in enumerate(channels):

        if not i+2 == len(channels):
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         ReLU())
             
        else:
            Gen.append(         ConvTranspose1d(ch,channels[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         Sigmoid())

            Gen.append(         Conv1d(channels[i+1],ch_1,kernel_size=kernel_1,stride=1,padding=int(kernel_1/2),bias=False))
            Gen.append(         BatchNorm1d(ch_1))
            Gen.append(         Sigmoid())

            Gen.append(         Conv1d(ch_1,ch_2,kernel_size=kernel_2,stride=1,padding=int(kernel_2/2),bias=False))
            Gen.append(         BatchNorm1d(ch_2))
            Gen.append(         Sigmoid())

            Gen.append(         Conv1d(ch_2,out_ch,kernel_size=kernel_3,stride=1,padding=int(kernel_3/2),bias=True))

            Gen.append(         Tanh())
            break
    
    return Gen.to(device)

def build_upsamp(ncz=512,out_ch=1,kernel_ver=0,factor_ver=0,leak=.2,device=torch.device('cuda')):
    factors     = [[7,7,5,5,4,2,2]][factor_ver]
    kernels     = [[[3,11,5],[7,65
                              ,5],[7,129,7],[7,256,9],[7,129,9],[7,129,9],[7,65,11],[7,33,5]]][kernel_ver]
    channels        = [512,256,256,256,128,96,64,48]  

    Gen     = Sequential(   ConvTranspose1d(ncz,channels[0],9,factors[0]),
                            BatchNorm1d(channels[0]),
                            LeakyReLU(negative_slope=.02,inplace=True))

    kernel_1        = 5 
    kernel_2        = 13
    kernel_3        = 7
    kernel_4        = 7
    kernel_5        = 5 
    kernel_6        = 3 


    ch_1            = 48
    ch_2            = 48
    ch_3            = 32
    ch_4            = 32
    ch_5            = 32

    cur_shape       = 9
    for i,ch in enumerate(channels):

        if not i+2 == len(channels):
            Gen.append(         Upsample(size=(factors[i]*cur_shape)))

            Gen.append(         Conv1d(channels[i],channels[i+1],kernel_size=kernels[i][0],stride=1,padding=int(kernels[i][0]/2),bias=False))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         LeakyReLU(negative_slope=leak,inplace=True))

            Gen.append(         Conv1d(channels[i+1],channels[i+1],kernel_size=kernels[i][1],stride=1,padding=int(kernels[i][1]/2),bias=False))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         LeakyReLU(negative_slope=leak,inplace=True))

            Gen.append(         Conv1d(channels[i+1],channels[i+1],kernel_size=kernels[i][2],stride=1,padding=int(kernels[i][2]/2),bias=False))
            Gen.append(         BatchNorm1d(channels[i+1]))
            Gen.append(         LeakyReLU(negative_slope=leak,inplace=True))

            
            #Gen.append(         BatchNorm1d(channels[i+1]))
            #Gen.append(         LeakyReLU())
             
        else:
            Gen.append(         Upsample(size=(factors[i]*cur_shape)))

            Gen.append(         Conv1d(channels[i],ch_1,kernel_size=kernel_1,stride=1,padding=int(kernel_1/2),bias=False))
            Gen.append(         BatchNorm1d(ch_1))
            Gen.append(         LeakyReLU(negative_slope=leak,inplace=True))

            Gen.append(         Conv1d(ch_1,ch_2,kernel_size=kernel_2,stride=1,padding=int(kernel_2/2),bias=False))
            Gen.append(         BatchNorm1d(ch_2))
            Gen.append(         LeakyReLU(negative_slope=leak,inplace=True))

            Gen.append(         Conv1d(ch_2,ch_3,kernel_size=kernel_3,stride=1,padding=int(kernel_3/2),bias=False))
            Gen.append(         BatchNorm1d(ch_3))
            Gen.append(         LeakyReLU(negative_slope=leak,inplace=True))

            #Gen.append(         Conv1d(ch_3,ch_4,kernel_size=kernel_4,stride=1,padding=int(kernel_4/2),bias=True))
            #Gen.append(         BatchNorm1d(ch_4))
            #Gen.append(         LeakyReLU(negative_slope=.5,inplace=True))

            #Gen.append(         Conv1d(ch_4,ch_5,kernel_size=kernel_5,stride=1,padding=int(kernel_5/2),bias=True))
            #Gen.append(         BatchNorm1d(ch_5))
            #Gen.append(         LeakyReLU(negative_slope=.5,inplace=True))

            Gen.append(         Conv1d(ch_5,out_ch,kernel_size=kernel_6,stride=1,padding=int(kernel_6/2),bias=True))
            Gen.append(         Tanh())
            break
            
        cur_shape *= factors[i]
    
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

#BEST 
def buildBest(ncz=512,leak=.2,kernel_ver=1,factor_ver=0,device=torch.device('cuda'),ver=1,out_ch=2):
    factors     = [[15,8,7,7,5,2,3],[2,5,7,7,8,9,15],[15,5,9,7,8,7,2]][factor_ver]

    ch          = [2048,2048,2048,512,256,128]

    ker         = [
                    [3,61,513,1025,129],
                    [3,5,9,17,65]][kernel_ver]

    pad         = [int(k/2) for k in ker] 
    Gen         = Sequential(   ConvTranspose1d(ncz,ch[0],factors[0],factors[0]),
                                BatchNorm1d(ch[0]),
                                LeakyReLU(leak,True))

    Gen.append(                 Conv1d(ch[0],ch[0],3,1,1))
    Gen.append(                 BatchNorm1d(ch[0]))
    Gen.append(                 LeakyReLU(leak,True)) 
    
    for i,c in enumerate(ch):
        if i+1 == len(ch):
            Gen.append(         ConvTranspose1d(c,128,factors[i+1],factors[i+1]))

            Gen.append(         Conv1d(128,64,factors[i+1]*3,1,int((factors[i+1]*3)/2)))
            Gen.append(         Sigmoid())
            Gen.append(         Conv1d(64,out_ch,factors[i+1],1,int((factors[i+1])/2)))
            Gen.append(         Tanh())

        else:
            Gen.append(         ConvTranspose1d(c,ch[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(ch[i+1])) 
            Gen.append(         LeakyReLU(leak,True)) 

            Gen.append(         Conv1d(ch[i+1],ch[i+1],ker[i],1,pad[i]))
            Gen.append(         BatchNorm1d(ch[i+1]))
            Gen.append(         LeakyReLU(leak,True)) 
    
    return Gen.to(device)

def buildBestMod1(ncz=512,leak=.2,kernel_ver=1,factor_ver=0,device=torch.device('cuda'),out_ch=2,verbose=False):
    factors     = [[15,8,7,7,5,2,3],[2,3,5,7,7,8,15],[15,5,9,7,8,7,2]][factor_ver]

    ch          = [ncz,ncz,int(ncz/2),int(ncz/2),64,48]


    Gen         = Sequential(   ConvTranspose1d(ncz,ch[0],factors[0],factors[0]),
                                BatchNorm1d(ch[0]),
                                LeakyReLU(leak,True))

    Gen.append(                 Conv1d(ch[0],ch[0],3,1,1))
    Gen.append(                 BatchNorm1d(ch[0]))
    Gen.append(                 LeakyReLU(leak,True)) 

    Gen.append(                 Conv1d(ch[0],ch[0],7,1,3))
    Gen.append(                 BatchNorm1d(ch[0]))
    Gen.append(                 LeakyReLU(leak,True)) 
    
    for i,c in enumerate(ch):
        
        
        if i+1 == len(ch):
            n_ch                = 48
            Gen.append(         ConvTranspose1d(c,n_ch,factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(n_ch))
            Gen.append(         LeakyReLU(leak,True)) 


            #ker_size            = 5 
            #n_ch_prev           = n_ch
            #n_ch                = 64
            #Gen.append(         Conv1d(n_ch_prev,n_ch,ker_size,1,int(ker_size/2),bias=False))
            #Gen.append(         BatchNorm1d(n_ch))
            #Gen.append(         LeakyReLU(leak,True)) 

            ker_size            = 13 
            n_ch_prev           = n_ch
            n_ch                = 64
            Gen.append(         Conv1d(n_ch_prev,n_ch,ker_size,1,int(ker_size/2),bias=True))
            Gen.append(         BatchNorm1d(n_ch))
            Gen.append(         LeakyReLU(leak,True)) 

            ker_size            = 5 
            n_ch_prev           = n_ch
            n_ch                = 64
            Gen.append(         Conv1d(n_ch_prev,n_ch,ker_size,1,int(ker_size/2),bias=True))
            Gen.append(         BatchNorm1d(n_ch))
            Gen.append(         LeakyReLU(leak,True)) 

            ker_size            = 5 
            n_ch_prev           = n_ch
            n_ch                = 64
            Gen.append(         Conv1d(n_ch_prev,n_ch,ker_size,1,int(ker_size/2),bias=False))
            Gen.append(         BatchNorm1d(n_ch))
            Gen.append(         LeakyReLU(leak,True)) 

            Gen.append(         Conv1d(n_ch,out_ch,3,1,1))
            Gen.append(         Tanh())

        else:
            Gen.append(         ConvTranspose1d(c,ch[i+1],factors[i+1],factors[i+1]))
            Gen.append(         BatchNorm1d(ch[i+1])) 
            Gen.append(         LeakyReLU(leak,True)) 

            ker_size            = 5 
            Gen.append(         Conv1d(ch[i+1],ch[i+1],ker_size,1,int(ker_size/2),bias=False))
            Gen.append(         BatchNorm1d(ch[i+1]))
            Gen.append(         LeakyReLU(leak,True)) 

            ker_size            = 21 if i < 3 else 13
            Gen.append(         Conv1d(ch[i+1],ch[i+1],ker_size,1,int((ker_size*1)/2),bias=False))
            Gen.append(         BatchNorm1d(ch[i+1]))
            Gen.append(         LeakyReLU(leak,True))
            
            ker_size            = 11 if i < 3 else 7 
            Gen.append(         Conv1d(ch[i+1],ch[i+1],ker_size,1,int((ker_size*1)/2),bias=False))
            Gen.append(         BatchNorm1d(ch[i+1]))
            Gen.append(         LeakyReLU(leak,True))  

            #ker_size            = 5 
            #Gen.append(         Conv1d(ch[i+1],ch[i+1],ker_size,1,int((ker_size*1)/2),bias=False))
            #Gen.append(         BatchNorm1d(ch[i+1]))
            #Gen.append(         LeakyReLU(leak,True)) 

    Gen     = Gen.to(device)

    if verbose:
        print(Gen)
    return Gen

if __name__ == "__main__":
    kernels     = [15,17,21,23,25,7,5]
    paddings    = [int(k/2) for k in kernels]
    D2      = AudioDiscriminator(channels=[2,32,64,128,256,512,1024,1],kernels=kernels,strides=[12,9,7,7,5,5,4],paddings=paddings,device=torch.device('cuda'),final_layer=1,verbose=False)
    inp = torch.randn((1,2,529200),device=torch.device('cuda'))

    print(f"shape: {D2.forward(inp).item()}")