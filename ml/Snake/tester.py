import time 
import random
import sys 
import os 
import subprocess
from matplotlib import pyplot as plt  
from trainer import Trainer  
import torch

if __name__ == "__main__":
    for lr in [.001,.0001,.00001]:
        t = Trainer(10,10,visible=False,loading=False,memory_size=2,loss_fn=torch.nn.MSELoss,architecture=[[6,32,3],[4608,128],[128,4]],gpu_acceleration=True,lr=lr)
        t.train_concurrent(iters=512,train_every=1024*2,memory_size=1024*16,sample_size=1024*8,batch_size=256,epochs=2)