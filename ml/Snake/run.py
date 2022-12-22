
from trainer import Trainer  
import torch
import sys 
from telemetry import plot_game
import numpy 
import copy 


variant_key = "x" 
arch_used   = 'None'
use_gpu = False

#ARCHITECTURES 
FCN_1 = {   "type":"FCN",
            "arch":[3,1024,128,4]}
#Short, med to small kernel _TOP
CNN_1 = {   "type":"CNN",
            "arch":[[3,16,7],[16,8,3],[6400,512],[512,4]]}
#Short, med to small kernel sequence
CNN_2 = {   "type":"CNN",
            "arch":[[3,16,9],[16,32,3],[12800,1024],[1024,4]]}
#Long Filter Sequence 
CNN_3 = {   "type":"CNN",
            "arch":[[3,8,11],[8,8,3],[6400,1024],[1024,4]]}



#LOSSES
HUBER   = torch.nn.HuberLoss
MSE     = torch.nn.MSELoss
MAE     = torch.nn.L1Loss

#OPTIMIZERS
ADAM    = torch.optim.Adam
ADAMW   = torch.optim.AdamW
ADA     = torch.optim.Adamax

#DICT ORGANIZER
MODELS = {  "FCN1" : FCN_1,
            "CNN1" : CNN_1,
            "CNN2" : CNN_2,
            "CNN3" : CNN_3
        }


#SETTINGS 
settings = {
    "x"     : 20,
    "y"     : 20,
    "lr"    : 5e-4,
    "it"    : 1024*512,
    "te"    : 256,
    "ps"    : 1024*64,
    "ss"    : 1024*16,
    "bs"    : 256,
    "ep"    : 1,
    "ms"    : 3,
    "mx"    : 200,
    "sf"    : 1,
    "arch"  : MODELS["CNN3"],
    "lo"    : HUBER,
    "op"    : ADAM,
    "tr"    : 4
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
print(variant_key)
if not isinstance(settings[variant_key],list):
    settings[variant_key] = [settings[variant_key]]
for setting in settings:
    if  setting == variant_key or setting == "sf":
        continue 
    v_0 = settings[setting]
    v_list = [copy.copy(v_0) for i in range( len(settings[variant_key]) if variant_key else 1)]
    settings[setting] = v_list


# RUN IT?
import pprint 
pprint.pp(settings)
if( not input(f"Proceed? [y/n]: ") in ["Y","y","Yes","yes"]): exit(0)


if __name__ == "__main__":

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

        #Correct model input 
        if "CNN" in model_type:
            model_arch[0][0] *= memory_size

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
                                                                        name=arch_used)

            scores,steps,best,x_scale,graph_name = t.train_concurrent(  iters=iters,
                                                                        train_every=train_every,
                                                                        pool_size=pool_size,
                                                                        sample_size=sample_size,
                                                                        batch_size=batch_size,
                                                                        epochs=epochs,
                                                                        max_steps=max_steps,
                                                                        blocker=256,
                                                                        transfer_models_every=transfer_rate) 

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
        print(f"max of {x_scale_lengths} is {numpy.argmax(x_scale_lengths)}")
        x_scale = x_scales[numpy.argmax(x_scale_lengths)]

        variant_sessions[str(settings[variant_key][v_i])]["avg_scores"] = avg_scores
        variant_sessions[str(settings[variant_key][v_i])]["avg_steps"]  = avg_steps
        variant_sessions[str(settings[variant_key][v_i])]["name"]       = name 
        variant_sessions[str(settings[variant_key][v_i])]["x_scale"]    = x_scale


    print([x for x in variant_sessions])
    plot_game(  scores_list     = [x["avg_scores"] for x in variant_sessions.values()],
                steps_list      = [x["avg_steps"] for x in variant_sessions.values()],
                series_names    = [f"{reverser[variant_key]} - {k}" for k in variant_sessions],
                x_scales        = [x["x_scale"] for x in variant_sessions.values()],
                graph_name      = big_name,
                f_name          = f"Results_vs_{variant_key}")