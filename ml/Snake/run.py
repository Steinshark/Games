
from trainer import Trainer  
import torch
import sys 
from telemetry import plot_game,ARCHITECTURES
import numpy 
import copy 
from torch.nn import Conv2d,ReLU,Flatten,Linear,MaxPool2d,Softmax, BatchNorm2d
from itertools import product
from matplotlib import pyplot as plt 
import random 
import os 
variant_keys    = []
arch_used       = 'None'
use_gpu         = False
sf              = 2
chunks          = 256
#ARCHITECTURES 
#LOSSES
HUBER   = torch.nn.HuberLoss
MSE     = torch.nn.MSELoss
MAE     = torch.nn.L1Loss

#OPTIMIZERS
ADAM    = torch.optim.Adam
ADAMW   = torch.optim.AdamW
ADA     = torch.optim.Adamax


#SETTINGS 
settings = {
    "x"     : 10,
    "y"     : 10,
    "lr"    : 25e-4,
    "it"    : 1024*64,
    "te"    : 64,
    "ps"    : 1024*8,
    "ss"    : 1024,
    "bs"    : 16,
    "ep"    : 1,
    "ms"    : 2,
    "mx"    : 200,
    "lo"    : MSE,
    "op"    : ADAMW,
    "tr"    : 16,
    "ga"    : .75,
    "rw"    : {"die":-.65,"eat":1.45,"step":0},
    "arch"  : ARCHITECTURES
}

reverser = {
    "x"     : "width",
    "y"     : "height",
    "lr"    : "learning rate",
    "it"    : "iterations",
    "te"    : "train every",
    "ps"    : "pool size",
    "ss"    : "sample size",
    "bs"    : "batch size",
    "ep"    : "epochs",
    "ms"    : "memory",
    "mx"    : "max steps",
    "sf"    : 1,
    "arch"  : "architecture",
    "lo"    : "loss",
    "tr"    :"transfer rate"
}


#ARG PARSER 
if len(sys.argv) > 1:
    i = 1 
    while True:
        try:

            #get key and val pair from command line 
            key = sys.argv[i]
            val = sys.argv[i+1]
            
            if key == "gpu":
                use_gpu == val in ["T","t"]
                i += 2
                continue
            if not key in settings:
                print("\n\nPlease choose from one of the settings:")
                print(list(settings.keys()))
                exit(0)


            settings[key] = eval(val)
            if isinstance(eval(val),list):
                variant_key = key
            if key == 'arch':
                arch_used = val 

            i += 2 
        except IndexError:
            break 


#PREPARATION
for setting in settings:
    if isinstance(settings[setting],list):
        variant_keys.append(setting)
        print(f"append {setting}")
        print(variant_keys)
        continue
    if setting == "sf":
        continue 
    settings[setting] = [settings[setting]]

print(variant_keys)

#Check for Compatability 
if len(variant_keys) > 3:
    print(f"Too many dimensions to test: {len(variant_keys)}, need 3")
    exit()

# RUN IT?

if __name__ == "__main__":
    import pprint 
    a = list(settings.values())
    all_settings = list(product(*a))
    if len(settings[variant_keys[1]]) < 2 and len(settings[variant_keys[0]]) > 1:
        variant_keys = [variant_keys[1],variant_keys[0],variant_keys[2]]
    FIG,PLOTS = plt.subplots(   nrows=len(settings[variant_keys[0]]),
                                ncols=len(settings[variant_keys[1]]))
    print(len(settings[variant_keys[0]]))
    print(len(settings[variant_keys[1]]))
    print(PLOTS.shape)

    i = 1

    print(f"x dim is {variant_keys[0]}")
    print(f"y dim is {variant_keys[1]}")
    for x,dim_1 in enumerate(settings[variant_keys[0]]):
        if dim_1 == "skip": 
            continue
        for y,dim_2 in enumerate(settings[variant_keys[1]]):
            if dim_2 == "skip":
                continue 
            
            for dim_3 in settings[variant_keys[2]]:
                if dim_3 == "skip":
                    continue
                

                t_scores = [0] * chunks
                t_steps  = [0] * chunks 
                x_scales = []
                name = f"{variant_keys[0]}-{str(dim_1)[:40]} x {variant_keys[1]}{str(dim_2)[:40]}"

                for iter in range(sf):

                    #PREP SETTINGS 
                    settings_dict = copy.deepcopy(settings)
                    for item in settings_dict:
                        if isinstance(settings_dict[item],list):
                            settings_dict[item] = settings_dict[item][0]
                    settings_dict[variant_keys[0]] = dim_1
                    settings_dict[variant_keys[1]] = dim_2
                    settings_dict[variant_keys[2]] = dim_3

                    if variant_keys[2] == 'arch':
                        series_name = list(ARCHITECTURES.keys())[list(ARCHITECTURES.values()).index(dim_3)]
                    else:
                        series_name = f"{variant_keys[2]}-{dim_3}"[:50]

                    #CORRECT ARCH 
                    print(f"Training iter\t{i}\{len(all_settings)}")
                    pprint.pp(settings_dict)
                    trainer = Trainer(  settings_dict['x'],settings_dict['y'],
                                        memory_size         =settings_dict['ms'],
                                        loss_fn             =settings_dict['lo'],
                                        optimizer_fn        =settings_dict['op'],
                                        lr                  =settings_dict['lr'],
                                        gamma               =settings_dict['ga'],
                                        architecture        =copy.deepcopy(settings_dict['arch']['arch']),
                                        gpu_acceleration    =use_gpu,
                                        epsilon             =.2,
                                        m_type              = settings_dict['arch']['type'],
                                        score_tracker       =list(),
                                        step_tracker        =list(),
                                        game_tracker        =list(),  
                                        gui=False     
                                        )
                    scores,steps,highscore,x_scale,xname = trainer.train_concurrent(    iters                   =settings_dict['it'],
                                                                                        train_every             =settings_dict['te'],
                                                                                        pool_size               =settings_dict['ps'],
                                                                                        batch_size              =settings_dict['bs'],
                                                                                        epochs                  =settings_dict['ep'],
                                                                                        transfer_models_every   =settings_dict['te'],
                                                                                        rewards                 =settings_dict['rw'],
                                                                                        max_steps               =settings_dict['mx'],
                                                                                        verbose=False)
                    
                    
                    #ADD ALL SCORES AND LIVES                                                                 pool_size               =settings_dict['ps'],
                    t_scores = [t + s for t,s in zip(scores,t_scores)]
                    t_steps  = [t + s for t,s in zip(steps,t_steps)]
                    x_scales.append(x_scale) 
                PLOTS[x][y].plot(x_scales[-1],t_scores,label=series_name)
                PLOTS[x][y].set_title("SCORES "+ name)
                PLOTS[x][y].legend()
                
                #Add to large figure 
                i+= 1

        
    plt.show()
    plt.savefig(os.join("figs",name))