import time 
import random
import sys 
import os 
import subprocess
from matplotlib import pyplot as plt  
from trainer import Trainer  


if __name__ == "__main__":
    t = Trainer(6,6,visible=False,loading=False,memory_size=2,architecture=[[6,16,3],[1024,32],[32,4]],gpu_acceleration=True,lr=.001)
    t.train_concurrent(iters=512,train_every=64,memory_size=1024*4,sample_size=128,batch_size=8,epochs=1)