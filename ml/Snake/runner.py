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
iters           = 1024*8
train_every     = 16
pool_size       = 2048 
sample_size     = 512 
bs              = 24
dr              = .2
minimum_thresh  = .03 
max_steps       = 100 
gamma           = [.75,.9,.98] 
rand_pick       = [0,.5]
kwargs          = {'weight_decay':.000001,'lr':.00001}
tr              = 10
ga              = .95
#DATA 
data            = {}
t0              = time.time()

repeats         = 1
if __name__ == "__main__":
    for die in [-1,-.9,-.8,-.7,-.5]:
        for eat in [2.5,1.5,1]:
            key     = f"die={die},eat={eat}"
            t1                              = time.time()
            for _ in range(repeats):
                t                               = Trainer(10,10,visible=False,loading=False,loss_fn=torch.nn.MSELoss,gpu_acceleration=True,gamma=ga,kwargs=kwargs,min_thresh=minimum_thresh,display_img=False,dropout_p=.25)
                reward          = {"die":die,"eat":eat,"step":0}

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
                                                                        verbose=True,
                                                                        x_scale=100,
                                                                        timeout=10*60)
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

    for key in data:
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