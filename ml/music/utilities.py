import torch 
import json 

D_CONFIGS = {
                "LoHi_1": {     'kernels': [4, 4, 16, 32, 128],
                                'strides': [8, 16, 16, 16, 64],
                                'paddings': [128, 8, 2, 16, 4]},

                "LoHi_2": {     'kernels': [16, 16, 512, 512, 4096, 4096],
                                'strides': [4, 4, 4, 4, 4, 16],
                                'paddings': [32, 2, 4, 1, 8, 1]},

                "LoHi_3":{      'kernels': [16, 32, 32, 32, 32, 32],
                                'strides': [4, 16, 16, 16, 16, 32],
                                'paddings': [2, 64, 4, 8, 128, 1]},

                "LoHi_4":{      'kernels': [16, 32, 32, 32, 32, 32],
                                'strides': [8, 8, 8, 16, 16, 32],
                                'paddings': [128, 64, 64, 1, 16, 2]},

                "LoHi_5":{      'kernels': [16, 16, 32, 32, 32, 32],
                                'strides': [2, 4, 16, 32, 32, 32],
                                'paddings': [8, 1, 8, 32, 128, 4]},

                "new"  : {      'kernels': [9,33,33,129,129,513,2049],
                                'strides': [3,3,3,3,3,3,3],
                                'paddings':[4,4,4,4,4,4,4],
                                'final_layer':1678},
                "new2" : {
                                'kernels' : [9,33,33,33,33],
                                'paddings' : [4,4,4,4,4],
                                'strides' : [7,5,5,4,4,3,3,3],
                                'final_layer' : 182}              
}

G_CONFIGS ={
                "HiLo_1" :{     'kernels': [4096, 4096, 128, 64, 64, 64, 64, 64, 4],
                                'strides': [2, 3, 2, 3, 3, 3, 1, 3, 2],
                                'paddings': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                'out_pad': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                'in_size': 1,
                                'num_channels': 2,
                                'channels': [100, 256, 128, 16, 16, 16, 16, 16, 4, 2],
                                'device': 'cuda'},

                "HiLo_2":{      'kernels': [4096, 2048, 2048, 128, 64, 64, 64, 64, 16],
                                'strides': [1, 2, 3, 3, 1, 2, 3, 3, 3],
                                'paddings': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                'out_pad': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                'in_size': 1,
                                'num_channels': 2},

                "LoHi_1": {     'kernels': [64, 8192, 8192, 8192, 16384, 16384, 16384, 32768],
                                'strides': [3, 1, 2, 2, 3, 3, 3, 3],
                                'paddings': [0, 0, 0, 0, 0, 0, 0, 0],
                                'out_pad': [0, 0, 0, 0, 0, 0, 0, 0],
                                'in_size': 1,
                                'num_channels': 2,
                                'channels': [100, 256, 128, 16, 16, 16, 16, 16, 4, 2],
                                'device': 'cuda'
                                },
                "USamp" : {
                                "factors"       : [2,3,3,3,4,4,5,5,5,7],
                                "channels"      : [1024,256,128,64,64,32,32,32,8,8,2],
                                "scales"        : [3,4,4,4,5,5,6,6,6,8]
                }
}

OPTIMIZER = {}
def weights_initG(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data,.2)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data,.3)
        torch.nn.init.constant_(m.bias.data, 0)

def weights_initD(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data,0,.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data,1,.02)
        torch.nn.init.constant_(m.bias.data, 0)

def config_explorer(configs,qualifications):

    #Store all valid configs
    passed = configs 

    #Check for all qualifications
    for qual in qualifications:
        passed = list(filter(qual,passed))

    if len(passed) == 0:
        print("No filters found")
        return [] 
    else:
        return passed 

def lookup(configs,config):
    for i,item in enumerate(configs):
        if item == config:
            return i 
    return -1 

def print_epoch_header(epoch_num,epoch_tot,header_width=100):
    print("="*header_width)
    epoch_seg   = f'=   EPOCH{f"{epoch_num+1}/{epoch_tot}".rjust(8)}'
    print(epoch_seg,end="")
    print(" "*(header_width-(len(epoch_seg)+1)),end="=\n")
    print("="*header_width)

def model_size(model:torch.nn.Module):
    return f"{ (sum([p.numel()*p.element_size() for p in model.parameters()])/(1024*1024)):.2f}MB"

if __name__ == "__main__":
    import numpy 
    a = numpy.load(r"C:\data\music\dataset\LOFI_sf5_t60\0f03823a5c_0.npy") 
    