import time 
import random


def runbuiltin():
    t0 = time.time() 

    y = 0
    for i in range(10000000):
        y = y + i 
    print(f"y equals {y} after {(time.time()-t0):.2f}s")


def runbetter():
    t0 = time.time() 

    y = 0
    for i in range(10000000):
        y = y.__add__(i)
    print(f"y equals {y} after {(time.time()-t0):.2f}s")


dirs = list(range(1000000))

def runmaybe():
    t0 = time.time() 

    for i in range(1000000):
        dirs[i] = max({
        (0,-1) 	: random.random(),
        (0,1) 	: random.random(),
        (-1,0) 	: random.random(),
        (1,0)	: random.random()
        },key={
        (0,-1) 	: random.random(),
        (0,1) 	: random.random(),
        (-1,0) 	: random.random(),
        (1,0)	: random.random()
        }.get)
    print(f"y equals after {(time.time()-t0):.2f}s")

    

runbuiltin() 
runbetter()
runmaybe()