
from trainer import Trainer  
import torch
import sys 
from telemetry import plot_game
import numpy 
import copy 
from torch.nn import Conv2d,ReLU,Flatten,Linear,MaxPool2d,Softmax, BatchNorm2d
from itertools import product
from matplotlib import pyplot as plt 
import random 

variant_keys    = []
arch_used       = 'None'
use_gpu         = False
sf              = 1
chunks          = 256
#ARCHITECTURES 
ARCHITECTURES = {   "smKer"     : {"type":"CNN","arch":[Conv2d(3,16,3,1,1) ,ReLU()          ,Conv2d(16,32,3,1,0),ReLU()                 ,Flatten()  ,Linear(512,128)    ,ReLU()             ,Linear(128,4)                                                  ]},
                    "medKer"    : {"type":"CNN","arch":[Conv2d(3,8,5,1,1)  ,ReLU()          ,Conv2d(8,16,3,1,0) ,ReLU()                 ,Flatten()  ,Linear(512,256)    ,ReLU()             ,Linear(256,4)                                                  ]},
                    "lrgKer"    : {"type":"CNN","arch":[Conv2d(3,8,13,1,1) ,ReLU()          ,Conv2d(8,16,3,1,0) ,ReLU()                 ,Flatten()  ,Linear(128,32)     ,ReLU()             ,Linear(32,4)                                                   ]},

                    "chatGPT"   : {"type":"CNN","arch":[Conv2d(3,32,5,1,2), BatchNorm2d(32) ,ReLU()             ,MaxPool2d(2, 2)        ,Conv2d(32,64,3,1,1)            ,BatchNorm2d(64)    ,ReLU()         ,MaxPool2d(2, 2)    ,Conv2d(64, 128, 3, 1, 1)   ,BatchNorm2d(128)   ,ReLU()     ,MaxPool2d(2, 2)    ,Flatten()  ,Linear(2048, 128)  ,ReLU() ,Linear(128, 4)]}
}

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
    "x"     : 12,
    "y"     : 12,
    "lr"    : 5e-4,
    "it"    : 1024*32,
    "te"    : 128,
    "ps"    : 1024*32,
    "ss"    : 1024*2,
    "bs"    : 32,
    "ep"    : 1,
    "ms"    : 2,
    "mx"    : 100,
    "arch"  : ARCHITECTURES,
    "lo"    : MSE,
    "op"    : ADAMW,
    "tr"    : 10,
    "ga"    : .79,
    "rw"    : {"die":-17,"eat":34,"step":1}
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

if __name__ == "__main__" and False :

    variant_sessions = {str(k) : {'avg_scores':[],"avg_steps":[],'name':[],'x_scale':[]} for k in settings[variant_key]}
    big_name = f"GRID[{settings['x'][0]},{settings['y'][0]}]-{arch_used}_[{set(settings['lo'])},Adam] @ {set(settings['lr'])} x [{settings['it'][0] * settings['te'][0]},{set(settings['te'])}] pool size {set(settings['ps'])} taking [{set(settings['ss'])},{set(settings['bs'])}]"

    for v_i in range(len(settings[variant_key])):
        grid_x          = settings["x"][v_i]
        grid_y          = settings["y"][v_i]
        memory_size     = settings["ms"][v_i]
        model_arch      = copy.deepcopy(settings["arch"][v_i]["arch"])
        learning_rate   = settings["lr"][v_i]
        model_type      = settings["arch"][v_i]["type"]
        iters           = settings['it'][v_i]
        train_every     = settings["te"][v_i]
        pool_size       = settings["ps"][v_i]
        sample_size     = settings["ss"][v_i]
        batch_size      = settings["bs"][v_i]
        epochs          = settings["ep"][v_i]
        max_steps       = settings["mx"][v_i]
        model_dict      = copy.deepcopy(settings["arch"][v_i])
        loss_fn         = settings["lo"][v_i]
        transfer_rate   = settings["tr"][v_i]

        if "CNN" in model_type:
            module  = model_arch    [0] 
            ch_in   = module.in_channels * memory_size
            ch_out  = module.out_channels
            pad     = module.padding
            kernel  = module.kernel_size
            stride  = module.stride
            model_arch[0] = Conv2d(ch_in,ch_out,kernel,stride,pad)
            
        elif "FCN" in model_type:
            model_arch[0] = memory_size * 3 * grid_x * grid_y


        score_list  = [] 
        steps_list  = [] 
        best_scores = []
        x_scales    = []
        name        = None 

        #   Run the iters of this settings variant batch 
        for i in range(settings['sf']):
            print(f"running with {model_arch}")
            t = Trainer(                                                grid_x,
                                                                        grid_y,
                                                                        visible=False,
                                                                        loading=False,
                                                                        memory_size=memory_size,
                                                                        architecture=model_arch,
                                                                        gpu_acceleration=use_gpu,
                                                                        lr=learning_rate,
                                                                        loss_fn=loss_fn,
                                                                        m_type=model_type,
                                                                        name=arch_used,
                                                                        score_tracker=[],
                                                                        step_tracker=[]
                                                                        )

            scores,steps,best,x_scale,graph_name = t.train_concurrent(  iters=iters,
                                                                        train_every=train_every,
                                                                        pool_size=pool_size,
                                                                        sample_size=sample_size,
                                                                        batch_size=batch_size,
                                                                        epochs=epochs,
                                                                        max_steps=max_steps,
                                                                        blocker=256,
                                                                        transfer_models_every=transfer_rate,
                                                                        verbose=True) 

            score_list      .append(scores)
            steps_list      .append(steps)
            best_scores     .append(best)
            name            = graph_name 
            x_scales        .append(x_scale)
        
        avg_scores = []
        for i in range(max([len(x) for x in score_list])):
            avg_scores      .append(0)
            existed         = 0 
            for sl in score_list:
                try:
                    avg_scores[-1]  += sl[i]
                    existed         += 1 
                except IndexError:
                    pass 

        avg_steps = []
        for i in range(max([len(x) for x in steps_list])):
            avg_steps       .append(0)
            existed         = 0 
            for sl in steps_list:
                try:
                    avg_steps[-1]   += sl[i]
                    existed         += 1 
                except IndexError:
                    pass 

        x_scale_lengths = [len(x) for x in x_scales]
        x_scale = x_scales[numpy.argmax(x_scale_lengths)]

        variant_sessions[str(settings[variant_key][v_i])]["avg_scores"] = avg_scores
        variant_sessions[str(settings[variant_key][v_i])]["avg_steps"]  = avg_steps
        variant_sessions[str(settings[variant_key][v_i])]["name"]       = name 
        variant_sessions[str(settings[variant_key][v_i])]["x_scale"]    = x_scale


    plot_game(  scores_list     = [x["avg_scores"] for x in variant_sessions.values()],
                steps_list      = [x["avg_steps"] for x in variant_sessions.values()],
                series_names    = [f"{reverser[variant_key]} - {k}" for k in variant_sessions],
                x_scales        = [x["x_scale"] for x in variant_sessions.values()],
                graph_name      = big_name,
                f_name          = f"Results_vs_{variant_key}")


if __name__ == "__main__":
    import pprint 
    a = list(settings.values())
    all_settings = list(product(*a))
    
    FIG,PLOTS = plt.subplots(   nrows=len(settings[variant_keys[0]])*2,
                                ncols=len(settings[variant_keys[1]]))

    print(PLOTS.shape)

    i = 1

    print(f"x dim is {variant_keys[0]}")
    print(f"y dim is {variant_keys[1]}")
    for x,dim_1 in enumerate(settings[variant_keys[0]]):
        for y,dim_2 in enumerate(settings[variant_keys[1]]):
            
            for dim_3 in settings[variant_keys[2]]:
                

                t_scores = [0] * chunks
                t_steps  = [0] * chunks 
                x_scales = []
                name = f"{variant_keys[0]}-{dim_1} x {variant_keys[1]}{dim_2}"

                for iter in range(sf):

                    #PREP SETTINGS 
                    settings_dict = copy.deepcopy(settings)
                    for item in settings_dict:
                        if isinstance(settings_dict[item],list):
                            settings_dict[item] = settings_dict[item][0]
                    settings_dict[variant_keys[0]] = dim_1
                    settings_dict[variant_keys[1]] = dim_2
                    settings_dict[variant_keys[2]] = dim_3

                    series_name = f"{variant_keys[2]}-{dim_3}"
                    #CORRECT ARCH 
                    print(f"Training iter\t{i}\{len(all_settings)}")
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
                PLOTS[2*x][y].plot(x_scales[-1],t_scores,label=series_name)
                PLOTS[2*x][y].set_title("SCORES "+ name)
                PLOTS[2*x][y].legend()
                PLOTS[2*x+1][y].plot(x_scales[-1],t_steps,label=series_name)
                PLOTS[2*x+1][y].set_title("STEPS " + name)
                PLOTS[2*x+1][y].legend()
                #Add to large figure 
                i+= 1

        
    plt.show()