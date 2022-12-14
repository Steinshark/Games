
from trainer import Trainer  
import torch
import sys 


#MODELS 
FCN_1 = {   "type":"FCN",
            "arch":[384,1024,128,4]}

CNN_1 = {   "type":"CNN",
            "arch":[[3,32,7],[32,16,3],[6400,256],[256,4]]}


#DICT ORGANIZER
MODELS = {"CNN1" : CNN_1,"FCN1" : FCN_1}


#SETTINGS 
settings = {
    "x"     : 20,
    "y"     : 20,
    "it"    : 2048*2,
    "te"    : 512,
    "ms"    : 1024*32,
    "ss"    : 1024*4,
    "bs"    : 32,
    "ep"    : 1,
    "me"    : 3,
    "arch"  : {"CNN" : CNN_1,"FCN":FCN_1}
}

#ARG PARSER 
if len(sys.argv) > 1:
    i = 1 
    while True:
        try:
            key = sys.argv[i]
            val = sys.argv[i+1]

            if not key in settings:
                print("\n\nPlease chose from one of the settings:")
                print(list(settings.keys()))
                exit(0)

            if not key == "arch":
                settings[key] = eval(val)
            else:
                settings[key] = MODELS[val]

            i += 2 
        except IndexError:
            break 



if "CNN" in settings['arch']['type']:
    settings['arch']['arch'][0][0] *= settings['me']
elif "FCN" in settings['arch']['type']:
    settings['arch']['arch'][0] = settings['me'] * 3 * settings['x'] * settings['y']



# RUN IT 
print(f"Running with settings:")
print()
import pprint 
pprint.pp(settings)
if( not input(f"Proceed? [y/n]: ") in ["Y","y","Yes","yes"]): exit(0)



if __name__ == "__main__":
    for lr in [.001]:
        t = Trainer(settings['x'],settings['y'],visible=False,loading=False,memory_size=settings['me'],loss_fn=torch.nn.MSELoss,architecture=settings['arch']["arch"],gpu_acceleration=True,lr=lr,m_type=settings['arch']["type"])
        t.train_concurrent(iters=settings["it"],train_every=settings["te"],memory_size=settings["ms"],sample_size=settings["ss"],batch_size=settings["bs"],epochs=settings["ep"])