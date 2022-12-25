import random 
import os 
from matplotlib import pyplot as plt
import torch 
from torch.nn import ReLU,MaxPool2d,Conv2d,Linear,Softmax,Flatten
from tkinter import BooleanVar
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

    name = os.path.join("figs",f"{f_name}.png")
    u = 0
    if os.path.exists(name):
        f_name = f_name + str(u)
        u += 1
        name = os.path.join("figs",f"{f_name}.png")

    fig.savefig(name)


# ARCHITECTURES = {"FCN_1"    : {"type":"FCN","arch":[3,1024,128,4]},
#                  "CNN_1"    : {"type":"CNN","arch":[[3,16,7],[16,8,3],[6400,512],[512,4]]},
#                  "CNN_2"    : {"type":"CNN","arch":[[3,16,9],[16,32,3],[12800,1024],[1024,4]]},
#                  "CNN_3"    : {"type":"CNN","arch":[[3,8,11],[8,8,3],[6400,1024],[1024,4]]},
#                  "CNN_4"    : {"type":"CNN","arch":[[3,32,3],[32,64,3],[128,64],[64,4]]}}

ARCHITECTURES = {"FCN_1"    : {"type":"FCN","arch":[3,1024,128,4]},
                 "CNN_1"    : {"type":"CNN","arch":[Conv2d(3,32,3,1,1),ReLU(),Conv2d(32,64,3,1,0),ReLU(),Flatten(),Linear(512,128),ReLU(),Linear(128,4)]},
                 "med"    : {"type":"CNN","arch":[Conv2d(3,8,11,1,1),ReLU(),Conv2d(8,8,3,1,1),ReLU(),Flatten(),Linear(512,128),ReLU(),Linear(128,4)]},

                 "CNN_2"    : {"type":"CNN","arch":[Conv2d(3,8,5,1,1),ReLU(),MaxPool2d(2,2),Conv2d(16,16,3,1,0),ReLU(),MaxPool2d(2,2),Flatten(),Linear(128,8),ReLU(),Linear(8,4)]},
                 "CNN_3"    : {"type":"CNN","arch":[Conv2d(3,16,11,1,1),ReLU(),MaxPool2d(2,2),Conv2d(16,16,3,1,0),ReLU(),MaxPool2d(2,2),Flatten(),Linear(128,16),ReLU(),Linear(16,4)]},
                 "CNN_4"    : {"type":"CNN","arch":[Conv2d(3,32,3,1,1),ReLU(),MaxPool2d(2,2),Conv2d(32,64,3,1,0),ReLU(),MaxPool2d(2,2),Flatten(),Linear(1024,32),ReLU(),Linear(32,4),Softmax()]},
                 "3WSM100"     : {"type":"CNN","arch":[Conv2d(3,16,11,1,1),ReLU(),MaxPool2d(2,2),Conv2d(16,16,3,1,0),ReLU(),MaxPool2d(2,2),Flatten(),Linear(128,16),ReLU(),Linear(16,4),Softmax()]},
}#,Softmax(dim=0)]}}

LOSSES      = { "Huber"     : torch.nn.HuberLoss,
                "L1"        : torch.nn.L1Loss,
                "MSE"       : torch.nn.MSELoss}

OPTIMIZERS  = { "Adam"      : torch.optim.Adam,
                "AdamW"     : torch.optim.AdamW,
                "SGD"       : torch.optim.SGD,
                "RMSProp"   : torch.optim.RMSprop}

DEFAULTS    = { "gameX"     : 12,
                "gameY"     : 12,
                "iters"     : 50000,
                "te"        : 100,
                "ps"        : 20000,
                "ss"        : 1000,
                "bs"        : 50,
                "lr"        : .0015,
                "ep"        : 1,
                "ms"        : 2,
                "mx"        : 50,
                "sf"        : 1,
                "arch"      : "",
                "lo"        : "",
                "op"        : "",
                "tr"        : 10,
                "gpu"       : False
                }
