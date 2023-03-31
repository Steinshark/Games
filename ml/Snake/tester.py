import time 
import random
import sys 
import os 
import subprocess
from matplotlib import pyplot as plt  
from trainerIMG import Trainer  
import torch
import utilities




FCN_1 = {"type":"FCN","arch":[384,1024,128,4]}
CNN_1 = {"type":"CNN","arch":[[9,32,7],[32,16,3],[6400,256],[256,4]]}

MODEL = CNN_1
if __name__ == "__main__":
    for lr in [.001]:
        t = Trainer(10,10,visible=False,loading=False,loss_fn=torch.nn.MSELoss,architecture=MODEL["arch"],gpu_acceleration=True,m_type=MODEL["type"])
        t.train_concurrent(iters=200,train_every=5,pool_size=64,sample_size=8,batch_size=8,epochs=1,verbose=True)
    print(f"TIMEMULT: {utilities.TIME_MULT}")