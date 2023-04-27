import random 
import os 
from matplotlib import pyplot as plt
import torch 
from torch.nn import ReLU,MaxPool2d,Conv2d,Linear,Softmax,Flatten, BatchNorm2d, LeakyReLU
from tkinter import BooleanVar
from networks import ConvolutionalNetwork
#Plot a list of scores and lives from a run of snake trainer

def plot_game(scores_list=[],steps_list=[],series_names="Empty",x_scales=[],graph_name="NoName",f_name="iterations"):

    colors = ["red","green","orange","cyan","black","mediumblue","chocolate","deepskyblue","gold"]
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(19.2,10.8)

    random.shuffle(colors) 
    for i,sc,li,x,na in zip([l for l in range(len(scores_list))],scores_list,steps_list,x_scales,series_names):
        axs[0].plot(x,sc,label=na,color=colors[i])
        axs[1].plot(x,li,label=na,color=colors[i])

    axs[0].legend()
    axs[0].set_title("Average Score")
    axs[1].legend()
    axs[1].set_title("Average Steps")

    axs[0].set_xlabel("Game Number")
    axs[0].set_ylabel("Score")
    axs[1].set_xlabel("Game Number")
    axs[1].set_ylabel("Steps Taken")
    fig.suptitle(graph_name)
    #Save fig to figs directory
    if not os.path.isdir("figs"):
        os.mkdir("figs")

    u = 0
    name = os.path.join("figs",f"{f_name}{u}.png")
    while os.path.exists(name):
        u += 1
        name = os.path.join("figs",f"{f_name}{u}.png")

    fig.savefig(name)
    print(f"saved to {name}")


# ARCHITECTURES = {"FCN_1"    : {"type":"FCN","arch":[3,1024,128,4]},
#                  "CNN_1"    : {"type":"CNN","arch":[[3,16,7],[16,8,3],[6400,512],[512,4]]},
#                  "CNN_2"    : {"type":"CNN","arch":[[3,16,9],[16,32,3],[12800,1024],[1024,4]]},
#                  "CNN_3"    : {"type":"CNmedN","arch":[[3,8,11],[8,8,3],[6400,1024],[1024,4]]},
#                  "CNN_4"    : {"type":"CNN","arch":[[3,32,3],[32,64,3],[128,64],[64,4]]}}

ARCHITECTURES = {   
                    "sm"        : {"type":"CNN","arch":[Conv2d(2,32,3,1,1),LeakyReLU(.2),Flatten(),Linear(512,128),LeakyReLU(.2),Linear(128,4)]},
                    "sm5"       : {"type":"CNN","arch":[Conv2d(2,32,5,1,1),LeakyReLU(.2),Flatten(),Linear(512,128),LeakyReLU(.2),Linear(128,4)]},
                    "med"       : {"type":"CNN","arch":[Conv2d(2,16,3,1,1),LeakyReLU(.2),Conv2d(16,32,5,1,0),LeakyReLU(.2),Flatten(),Linear(512,128),LeakyReLU(.2),Linear(128,4)                         ]},
                    "med7"      : {"type":"CNN","arch":[Conv2d(2,64,3,1,1),LeakyReLU(.2),Conv2d(64,64,7,1,0),LeakyReLU(.2),Flatten(),Linear(512,128),LeakyReLU(.2),Linear(128,4),                         ]},
                    "lg"        : {"type":"CNN","arch":[Conv2d(2,16,3,1,1),LeakyReLU(.2),Conv2d(16,64,5,1,0),LeakyReLU(.2),Conv2d(64,64,5,1,0),LeakyReLU(.2),Flatten(),Linear(1,512),LeakyReLU(.2),Linear(512,64),LeakyReLU(.2) ,Linear(64,4)                       ]},
                    "chatGPT"   : {"type":"CNN","arch":[Conv2d(2,32,3,1,2),BatchNorm2d(32) ,LeakyReLU(.2),MaxPool2d(2, 2),Conv2d(32,64,3,1,1),BatchNorm2d(64),LeakyReLU(.2),MaxPool2d(2,2),Conv2d(64, 128, 3, 1, 1),BatchNorm2d(128)   ,LeakyReLU(.2)     ,MaxPool2d(2, 2)    ,Flatten()  ,Linear(2048, 128)  ,LeakyReLU(.2) ,Linear(128, 4) ]},
                    "new"       : {"type":"CNN","arch":[Conv2d(2,32,3,1,1),LeakyReLU(.2),Conv2d(32,64,5,1,1),LeakyReLU(.2),Conv2d(64,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Flatten(),Linear(16,4)]},
                    "newsm"     : {"type":"CNN","arch":[Conv2d(2,32,3,1,1),LeakyReLU(.2),Conv2d(32,64,5,1,1),LeakyReLU(.2),Conv2d(64,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Conv2d(32,32,5,1,1),LeakyReLU(.2),Flatten(),Linear(8,4)]}

}#,Softmax(dim=0)]}}

LOSSES      = { "Huber"     : torch.nn.HuberLoss,
                "L1"        : torch.nn.L1Loss,
                "MSE"       : torch.nn.MSELoss}

OPTIMIZERS  = { "Adam"      : torch.optim.Adam,
                "AdamW"     : torch.optim.AdamW,
                "SGD"       : torch.optim.SGD,
                "RMSProp"   : torch.optim.RMSprop}

DEFAULTS    = { "gameX"     : 20,
                "gameY"     : 20,
                "iters"     : 1024*32,
                "te"        : 8,
                "ps"        : 1024*16,
                "ss"        : 2048,
                "bs"        : 64,
                "lr"        : .0001,
                "kw"        : "{'weight_decay':.25e-5}",
                "ll"        : "[(-1,2.5e-5)]",
                "ep"        : 1,
                "mt"        : .03,
                "mx"        : 150,
                "sf"        : 1,
                "arch"      : "",
                "lo"        : "",
                "op"        : "",
                "tr"        : 3,
                "drop"      : .25,
                "gam"       : .9,
                "gpu"       : True,
                "rew"       : "{'die':-.75,'eat':1.6,'step':0}",
                "rpick"     : .3
                }



POTENTIAL   = { "gameX"     : 14,
                "gameY"     : 14,
                "iters"     : 1024*32,
                "te"        : 8,
                "ps"        : 1024*8,
                "ss"        : 1024,
                "bs"        : 32,
                "lr"        : .0001,
                "kw"        : "{'weight_decay':.5e-5}",
                "ll"        : "[(-1,3e-3),(256,5e-5)]",
                "ep"        : 1,
                "mt"        : .03,
                "mx"        : 125,
                "sf"        : 1,
                "arch"      : "",
                "lo"        : "",
                "op"        : "",
                "tr"        : 12,
                "drop"      : .25,
                "gam"       : .9,
                "gpu"       : True,
                "rew"       : "{'die':-.75,'eat':1.3,'step':0}",
                "rpick"     : .45
                }


# in_v    = torch.randn(size=(1,2,20,20),dtype=torch.float)

# model = ConvolutionalNetwork(torch.nn.MSELoss,torch.optim.Adam,lr=.0001,architecture=ARCHITECTURES['new']['arch'],input_shape=(1,2,20,20))

# print(f"Model Out: {model.forward(in_v).shape}")