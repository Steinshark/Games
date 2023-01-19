import torch 
from torch.nn import ConvTranspose1d,Upsample,BatchNorm1d,ReLU,LeakyReLU,Conv1d,Sequential,Tanh
from networks import AudioGenerator,AudioDiscriminator

from utilities import model_size
import math
from torch.nn import Upsample

#base_factors = [2,2,2,2,2,3,3,3,5,5,5,7,7] 
#           IDEA 1 -
# Use a ConvTranspose with strides equal to factors of input    
#ncz = 512 
#random_input    = torch.randn(size=(1,ncz,1))




#EXTRACT AUDIO FEATURES FIRST, THEN GO BACK UP 
def build_encdec(   in_factors      =[],
                    enc_factors     =[],
                    dec_factors     =[],
                    dec_channels    = [512,256,128,64,64,2],
                    dec_kernels     = [5,17,65,129,257,257,513,1025],
                    bs=8,
                    leak=.2):

    #Constants 
    factors         = [2,2,2,2,3,3,3,5,5,7,7]
    output          = 529_200

    #Start with 2 encoder layers
    enc_kernels     = [5,5,17]
    enc_channels    = [32,64,128]
    enc_strides     = enc_factors
    enc_padding     = [math.floor(k/2) for k in enc_kernels]

    #   ENCODER 
    G   = Sequential(   Conv1d(1,               enc_channels[0],enc_kernels[0],stride=enc_strides[0],padding=enc_padding[0]),
                        BatchNorm1d(enc_channels[0]),
                        LeakyReLU(leak,True),

                        Conv1d(enc_channels[0], enc_channels[1],enc_kernels[1],stride=enc_strides[1],padding=enc_padding[1]),
                        BatchNorm1d(enc_channels[1]),
                        LeakyReLU(leak,True),

                        Conv1d(enc_channels[1], enc_channels[2],enc_kernels[2],stride=enc_strides[2],padding=enc_padding[2]),
                        BatchNorm1d(enc_channels[2]),
                        LeakyReLU(leak,True),
                        )


    #Finish with decoder layers 
        # dec_channels    = [1024,512,512,256,128,64,32,32]
        # dec_kernels     = [9,9,17,17,17,17,17,17]
        # dec_strides     = dec_factors
        # dec_padding     = [0 for k in dec_kernels]
        #                 UPTO --------------]
    dec_padding     = [ math.floor(k/2) for k in dec_kernels]

    #G.append(           ConvTranspose1d(enc_channels[2],dec_channels[0],dec_kernels[0],stride=dec_strides[0],padding=dec_padding[0]))
    G.append(           Upsample(scale_factor=dec_factors[0]*2))
    G.append(           Conv1d(enc_channels[2],dec_channels[0],dec_kernels[0],stride=2,padding=dec_padding[0]))
    G.append(           BatchNorm1d(dec_channels[0]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[0],dec_channels[1],dec_kernels[1],stride=dec_strides[1],padding=dec_padding[1]))
    G.append(           Upsample(scale_factor=dec_factors[1]*2))
    G.append(           Conv1d(dec_channels[0],dec_channels[1],dec_kernels[1],stride=2,padding=dec_padding[1]))
    G.append(           BatchNorm1d(dec_channels[1]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[1],dec_channels[2],dec_kernels[2],stride=dec_strides[2],padding=dec_padding[2]))
    G.append(           Upsample(scale_factor=dec_factors[2]*2))
    G.append(           Conv1d(dec_channels[1],dec_channels[2],dec_kernels[2],stride=2,padding=dec_padding[2]))
    G.append(           BatchNorm1d(dec_channels[2]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[2],dec_channels[3],dec_kernels[3],stride=dec_strides[3],padding=dec_padding[3]))
    G.append(           Upsample(scale_factor=dec_factors[3]*2))
    G.append(           Conv1d(dec_channels[2],dec_channels[3],dec_kernels[3],stride=2,padding=dec_padding[3]))
    G.append(           BatchNorm1d(dec_channels[3]))
    G.append(           LeakyReLU(leak,True))
    
    #G.append(           ConvTranspose1d(dec_channels[3],dec_channels[4],dec_kernels[4],stride=dec_strides[4],padding=dec_padding[4]))
    G.append(           Upsample(scale_factor=dec_factors[4]*2))
    G.append(           Conv1d(dec_channels[3],dec_channels[4],dec_kernels[4],stride=2,padding=dec_padding[4]))
    G.append(           BatchNorm1d(dec_channels[4]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[4],dec_channels[5],dec_kernels[5],stride=dec_strides[5],padding=dec_padding[5]))
    G.append(           Upsample(scale_factor=dec_factors[5]*2))
    G.append(           Conv1d(dec_channels[4],dec_channels[5],dec_kernels[5],stride=2,padding=dec_padding[5]))
    # G.append(           BatchNorm1d(dec_channels[5]))
    # G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[5],dec_channels[6],dec_kernels[6],stride=dec_strides[6],padding=dec_padding[6]))
    # G.append(           Upsample(scale_factor=dec_strides[6]*2))
    # G.append(           Conv1d(dec_channels[5],dec_channels[6],dec_kernels[6],stride=2,padding=dec_padding[6]))
    G.append(           Tanh())

    # G.append(           ConvTranspose1d(dec_channels[6],dec_channels[7],dec_kernels[7],stride=dec_strides[7],padding=dec_padding[7]))
    # G.append(           BatchNorm1d(dec_channels[7]))
    # G.append(           LeakyReLU(leak,True))

    return G.to(torch.device('cuda'))

def build_encdec_multi(     in_factors      =[],
                            enc_factors     =[],
                            dec_factors     =[],
                            dec_channels    = [512,256,128,64,64,2],
                            dec_kernels     = [5,17,65,129,257,257,513,1025],
                            bs=8,
                            leak=.2):

    #Constants 
    factors         = [2,2,2,2,3,3,3,5,5,7,7]
    output          = 529_200

    #Start with 2 encoder layers
    enc_kernels     = [5,5,17]
    enc_channels    = [32,64,128]
    enc_strides     = enc_factors
    enc_padding     = [math.floor(k/2) for k in enc_kernels]

    #   ENCODER 
    G   = Sequential(   Conv1d(1,               enc_channels[0],enc_kernels[0],stride=enc_strides[0],padding=enc_padding[0]),
                        BatchNorm1d(enc_channels[0]),
                        LeakyReLU(leak,True),

                        Conv1d(enc_channels[0], enc_channels[1],enc_kernels[1],stride=enc_strides[1],padding=enc_padding[1]),
                        BatchNorm1d(enc_channels[1]),
                        LeakyReLU(leak,True),

                        Conv1d(enc_channels[1], enc_channels[2],enc_kernels[2],stride=enc_strides[2],padding=enc_padding[2]),
                        BatchNorm1d(enc_channels[2]),
                        LeakyReLU(leak,True),
                        )


    #Finish with decoder layers 
        # dec_channels    = [1024,512,512,256,128,64,32,32]
        # dec_kernels     = [9,9,17,17,17,17,17,17]
        # dec_strides     = dec_factors
        # dec_padding     = [0 for k in dec_kernels]
        #                 UPTO --------------]
    dec_padding     = [ math.floor(k/2) for k in dec_kernels]

    #G.append(           ConvTranspose1d(enc_channels[2],dec_channels[0],dec_kernels[0],stride=dec_strides[0],padding=dec_padding[0]))
    G.append(           Upsample(scale_factor=dec_factors[0]*2))
    G.append(           Conv1d(enc_channels[2],dec_channels[0],dec_kernels[0]*2,    stride=1,padding=dec_padding[0]*2,bias=False))
    G.append(           Conv1d(enc_channels[2],dec_channels[0],dec_kernels[0],      stride=2,padding=dec_padding[0],bias=False))
    G.append(           BatchNorm1d(dec_channels[0]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[0],dec_channels[1],dec_kernels[1],stride=dec_strides[1],padding=dec_padding[1]))
    G.append(           Upsample(scale_factor=dec_factors[1]*2))
    G.append(           Conv1d(dec_channels[0],dec_channels[1],dec_kernels[1]*2,    stride=1,padding=dec_padding[1]*2,bias=False))
    G.append(           Conv1d(dec_channels[0],dec_channels[1],dec_kernels[1],      stride=2,padding=dec_padding[1],bias=False))
    G.append(           BatchNorm1d(dec_channels[1]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[1],dec_channels[2],dec_kernels[2],stride=dec_strides[2],padding=dec_padding[2]))
    G.append(           Upsample(scale_factor=dec_factors[2]*2))
    G.append(           Conv1d(dec_channels[1],dec_channels[2],dec_kernels[2],      stride=1,padding=dec_padding[2]*2,bias=False))
    G.append(           Conv1d(dec_channels[1],dec_channels[2],dec_kernels[2],      stride=2,padding=dec_padding[2],bias=False))

    G.append(           BatchNorm1d(dec_channels[2]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[2],dec_channels[3],dec_kernels[3],stride=dec_strides[3],padding=dec_padding[3]))
    G.append(           Upsample(scale_factor=dec_factors[3]*2))
    G.append(           Conv1d(dec_channels[2],dec_channels[3],dec_kernels[3],      stride=1,padding=dec_padding[3]*2,bias=False))
    G.append(           Conv1d(dec_channels[2],dec_channels[3],dec_kernels[3],      stride=2,padding=dec_padding[3],bias=False))
    G.append(           BatchNorm1d(dec_channels[3]))
    G.append(           LeakyReLU(leak,True))
    
    #G.append(           ConvTranspose1d(dec_channels[3],dec_channels[4],dec_kernels[4],stride=dec_strides[4],padding=dec_padding[4]))
    G.append(           Upsample(scale_factor=dec_factors[4]*2))
    G.append(           Conv1d(dec_channels[3],dec_channels[4],dec_kernels[4],      stride=1,padding=dec_padding[4]*2,bias=False))
    G.append(           Conv1d(dec_channels[3],dec_channels[4],dec_kernels[4],      stride=2,padding=dec_padding[4],bias=False))
    G.append(           BatchNorm1d(dec_channels[4]))
    G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[4],dec_channels[5],dec_kernels[5],stride=dec_strides[5],padding=dec_padding[5]))
    G.append(           Upsample(scale_factor=dec_factors[5]*2))
    G.append(           Conv1d(dec_channels[4],dec_channels[5],dec_kernels[5],      stride=1,padding=dec_padding[5]*2,bias=False))
    G.append(           Conv1d(dec_channels[4],dec_channels[5],dec_kernels[5],      stride=2,padding=dec_padding[5],bias=False))
    # G.append(           BatchNorm1d(dec_channels[5]))
    # G.append(           LeakyReLU(leak,True))

    #G.append(           ConvTranspose1d(dec_channels[5],dec_channels[6],dec_kernels[6],stride=dec_strides[6],padding=dec_padding[6]))
    # G.append(           Upsample(scale_factor=dec_strides[6]*2))
    # G.append(           Conv1d(dec_channels[5],dec_channels[6],dec_kernels[6],stride=2,padding=dec_padding[6]))
    G.append(           Tanh())

    # G.append(           ConvTranspose1d(dec_channels[6],dec_channels[7],dec_kernels[7],stride=dec_strides[7],padding=dec_padding[7]))
    # G.append(           BatchNorm1d(dec_channels[7]))
    # G.append(           LeakyReLU(leak,True))

    return G.to(torch.device('cuda'))



if __name__ == "__main__":
    insize      = 1008*2

    g = build_encdec(in_factors=[2,2,2,3,3,3,7],enc_factors=[2,3,7],dec_factors=[3,3,5,5,7,7],bs=8)

    print(g)
    print(model_size(g))

    rands = torch.randn(size=(1,1,insize),device=torch.device("cuda"))
    print(g(rands).shape)

