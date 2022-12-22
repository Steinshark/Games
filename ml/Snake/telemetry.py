import random 
import os 
from matplotlib import pyplot as plt


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
