import time 
import random
import sys 
import os 
import subprocess
from matplotlib import pyplot as plt  
from trainerIMG import Trainer  
import torch
import utilities
import json 

#SETTINGS 
iters           = 1024*4
train_every     = 16 
pool_size       = 2048 
sample_size     = 1024 
bs              = 32 
minimum_thresh  = .03 
max_steps       = 100 
transfer_rate   = 16 
gamma           = .65 
rand_pick       = .1
kwargs          = {'weight_decay':.00001,'lr':.00001}

rlist           = [1.5,1.2]
trlist          = [2,64 ]
#DATA 
data            = {f"eat={x},tr={tr}" : None for x in rlist for tr in [2,16]}
t0              = time.time()
if __name__ == "__main__":
    for rw in rlist:
        for tr in trlist:
            t1                              = time.time()
            reward                          = {"die":-.85,"eat":rw,"step":0}
            t                               = Trainer(10,10,visible=False,loading=False,loss_fn=torch.nn.MSELoss,gpu_acceleration=True,gamma=gamma,kwargs=kwargs)

            scores,lived,high,gname = t.train_concurrent(           iters=iters,
                                                                    train_every=train_every,
                                                                    pool_size=pool_size,
                                                                    sample_size=sample_size,
                                                                    batch_size=bs,
                                                                    epochs=1,
                                                                    transfer_models_every=tr,
                                                                    rewards=reward,
                                                                    max_steps=max_steps,
                                                                    drop_rate=rand_pick,
                                                                    verbose=False,
                                                                    x_scale=100)
            data[f"eat={rw},tr={tr}"] = [list(scores),list(lived)]
            print(f"\tFinished {rw} in {(time.time()-t1):.2f}s")

    with open("datadump.txt","w") as file:
        file.write(json.dumps(data))
    

    #Split plots 
    fig,axs     = plt.subplots(nrows=2)

    for rw in rlist:
        for tr in trlist:
            name = f"eat={rw},tr={tr}"
            axs[0].plot(data[name][0],label=name)
            axs[1].plot(data[name][1],label=name)
    
    axs[0].set_xlabel("Generation")
    axs[1].set_xlabel("Generation")

    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel("Avg. Score")
    axs[1].set_ylabel("Avg. Survived")

    print(f"ran in : {(time.time()-t0):.2f}s")
    plt.show()