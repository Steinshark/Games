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
train_every     = 8
pool_size       = 2048 
sample_size     = 512 
bs              = 16 
minimum_thresh  = .03 
max_steps       = 100 
gamma           = .95 
rand_pick       = [0,.75]
kwargs          = {'weight_decay':.00001,'lr':.00001}

rlist           = [100,350]
trlist          = [4]
#DATA 
data            = {f"eat={x},tr={tr}" : None for x in rlist for tr in [2,16]}
t0              = time.time()
reward          = {"die":-1,"eat":1,"step":0}


repeats         = 2
if __name__ == "__main__":
    for tr in trlist:
        for dr in rand_pick:
            key     = f"tr={tr},dr={dr}"
            t1                              = time.time()
            for _ in range(repeats):
                t                               = Trainer(10,10,visible=False,loading=False,loss_fn=torch.nn.MSELoss,gpu_acceleration=True,gamma=gamma,kwargs=kwargs,min_thresh=minimum_thresh,display_img=False)

                scores,lived,high,gname = t.train_concurrent(           iters=iters,
                                                                        train_every=train_every,
                                                                        pool_size=pool_size,
                                                                        sample_size=sample_size,
                                                                        batch_size=bs,
                                                                        epochs=1,
                                                                        transfer_models_every=tr,
                                                                        rewards=reward,
                                                                        max_steps=max_steps,
                                                                        drop_rate=dr,
                                                                        verbose=False,
                                                                        x_scale=100,
                                                                        timeout=5*60)
                if not key in data:
                    data[key] = [list(scores),list(lived)]
                else:
                    data[key] = [[list(scores)[i] + data[key][0][i] for i in range(len(scores))],[list(lived)[i] + data[key][1][i] for i in range(len(scores))]]
            data[key] = [[i/repeats for i in data[key][0]],[i/repeats for i in data[key][1]]]
            print(f"\tFinished {key} in {(time.time()-t1):.2f}s\ths={high}\tlived={[f'{i:.2f}' for i in utilities.reduce_arr(lived,8)]}")

    with open("datadump.txt","w") as file:
        file.write(json.dumps(data))
    

    #Split plots 
    fig,axs     = plt.subplots(nrows=2)

    for tr in trlist:
        for dr in rand_pick:
            key = f"tr={tr},dr={dr}"
            axs[0].plot(data[key][0],label=key)
            axs[1].plot(data[key][1],label=key)
    
    axs[0].set_xlabel("Generation")
    axs[1].set_xlabel("Generation")

    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel("Avg. Score")
    axs[1].set_ylabel("Avg. Survived")

    print(f"ran in : {(time.time()-t0):.2f}s")
    plt.show()