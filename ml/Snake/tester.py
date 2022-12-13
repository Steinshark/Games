import time 
import random
import sys 
import os 
import subprocess
from matplotlib import pyplot as plt  
from trainer import Trainer  
import torch




FCN_1 = {"type":"FCN","arch":[384,1024,128,4]}
CNN_1 = {"type":"CNN","arch":[[9,32,7],[32,16,3],[6400,256],[256,4]]}

MODEL = CNN_1
if __name__ == "__main__":
    for lr in [.001]:
        t = Trainer(20,20,visible=False,loading=False,memory_size=3,loss_fn=torch.nn.MSELoss,architecture=MODEL["arch"],gpu_acceleration=True,lr=lr,m_type=MODEL["type"])
        t.train_concurrent(iters=2048*2,train_every=512,memory_size=1024*32,sample_size=512*16,batch_size=128,epochs=1)