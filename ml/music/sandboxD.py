import torch 
from torch.nn import ConvTranspose1d,Upsample,BatchNorm1d,ReLU,LeakyReLU,Conv1d,Sequential,Tanh
from networks import AudioGenerator,AudioDiscriminator
import math
from utilities import model_size
from torch.nn import Upsample


models = {
    "test"  : {
        'kernels'   :[],
        'padding'   :[],
        'srides'    :[]
                }
    
}


#INPUT IS NOW SIZE 
 

input_size  = 100
input_vect = torch.randn(size=(1,2,529_200),dtype=torch.float,device=torch.device('cuda'))

kernels     = [17,17,17,17,17,17,13,9,5]
paddings    = [math.floor(k/2) for k in kernels]
strides     = [7,7,5,5,4,4,3,3,3]



a = Sequential()

for i in range(len(kernels)):
    if i == 0:
        a.append(Conv1d(2,1,kernels[i],strides[i],paddings[i]))
    else:
        a.append(Conv1d(1,1,kernels[i],strides[i],paddings[i]))


a  = AudioDiscriminator(channels=[2,8,32,64,256,512,1024,2048,2048,1],kernels=kernels,strides=[7,7,5,5,4,4,3,3,3],paddings=paddings,device=torch.device('cuda'),final_layer=1,verbose=True)
a.to(torch.device('cuda'))

print("MODEL:")
print("\tout:",a(input_vect).shape)
print("\tsize:",sum([p.numel()*p.element_size() for p in a.parameters()])//(1024*1024),"MB")