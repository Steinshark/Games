from matplotlib import pyplot as plt 
import json 



if __name__ == "__main__":
    #Grab data from file 
    f = open("saved_states.txt","r").read()
    out_list = json.loads(f)
    data = {}

    graph_cats = ['lr','']


    #Prep the charts
    fig,axs = plt.subplots(2)

    for outcome in out_list:
        i = int(outcome['lr'] == 1e-6)
        axs[i].plot(outcome['avg_scores'],label=outcome['optimizer_fn'])
    
    fig.suptitle("Average RL Agent Snake Score ")
    axs[0].legend()
    axs[0].set_title("Learning Rate = 1e-3")
    axs[0].set_xlabel("Expisode (% / 75k)")
    axs[0].set_ylabel("Average Score")
    axs[1].legend()
    axs[1].set_title("Learning Rate = 1e-6")
    axs[1].set_xlabel("Expisode (% / 75k)")
    axs[1].set_ylabel("Average Score")
    plt.show()

