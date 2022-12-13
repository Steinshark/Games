import time 
import random
import sys 
import os 
import subprocess
from matplotlib import pyplot as plt  
from trainer import Trainer  
import torch




FCN_1 = {"type":"FCN","arch":[384,1024,128,4]}
CNN_1 = {"type":"CNN","arch":[[6,64,3],[64,8,3],[1152,512],[512,128],[128,4]]}

MODEL = CNN_1
if __name__ == "__main__":
    for lr in [.001]:#:,.0001,.00001]:
        t = Trainer(20,20,visible=False,loading=False,memory_size=3,loss_fn=torch.nn.MSELoss,architecture=MODEL["arch"],gpu_acceleration=True,lr=lr,m_type=MODEL["type"])
        t.train_concurrent(iters=64,train_every=2048,memory_size=1024*64,sample_size=1024*8,batch_size=64,epochs=2)