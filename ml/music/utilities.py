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
                                'paddings': [8, 1, 8, 32, 128, 4]}
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
                                }
}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
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


if __name__ == "__main__":
    configs     = json.loads(open("configs3.txt").read())
    d_configs   = list({str(con['kernels'])+str(con['strides'])+str(con['paddings']) : con for i,con in enumerate(configs['D'])}.values())
    g_configs   = list({str(con['kernels'])+str(con['strides'])+str(con['paddings']) : con for i,con in enumerate(configs['G'])}.values())

    print(len(d_configs))
    import pprint 
    conf = config_explorer(d_configs,[lambda con: con['kernels'][0] < 32000 and con['kernels'][0] >= 32])
    pprint.pp(conf[:10])
    pcik = int(input("pick: "))
    print(lookup(d_configs,conf[pcik]))
    pprint.pp(d_configs[lookup(d_configs,conf[pcik])])