from matplotlib import pyplot as plt 
import json 
import sys 


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-b":
            olist = json.loads(open(sys.argv[2],"r").read())
            fig,axs = plt.subplots(2)
            axs[0].plot(olist[-1])
            axs[1].plot(olist[-2])
            plt.show()
            exit()
    #Grab data from file 
    f = open("saved_states.txt","r").read()
    outlist = json.loads(f)
    data = {}

    graph_series = 'optimizer_fn'

    #Unique values of the series (in one box)
    series_vals = {l[graph_series] : None for l in outlist}

    #Rows we are tracking
    graph_rows = {'lr':0,'batch_size':0}
    
    #Num unique values for each row 
    for dim in graph_rows:
        graph_rows[dim] = {l[dim] for l in outlist}


    row_len = max([len(k) for k in graph_rows.values()])


    row_series = {k : {l : {i : None for i in series_vals} for l in graph_rows[k]} for k in graph_rows}

    import pprint 
    pprint.pp(row_series)
    input()

    #Prep the charts
    fig,axs = plt.subplots(nrows=len(graph_rows),ncols=max(graph_rows.values()))




    for outcome in outlist:
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

