import time 
import random
import sys 
import os 
import subprocess
from matplotlib import pyplot as plt  

# def printres(name,dt):
#     print(f"ran {name} in {(dt):.2f}s")

# def runbuiltin():
#     t0 = time.time() 

#     y = 0
#     for i in range(10000000):
#         y = y + i 
#     print(f"y equals {y} after {(time.time()-t0):.2f}s")


# def runbetter():
#     t0 = time.time() 

#     y = 0
#     for i in range(10000000):
#         y = y.__add__(i)
#     print(f"y equals {y} after {(time.time()-t0):.2f}s")



# def runmaybe():
#     dirs = list(range(1000000))
#     t0 = time.time() 
#     for i in range(1000000):
#         dirs[i] = max({
#         (0,-1) 	: random.random(),
#         (0,1) 	: random.random(),
#         (-1,0) 	: random.random(),
#         (1,0)	: random.random()
#         },key={
#         (0,-1) 	: random.random(),
#         (0,1) 	: random.random(),
#         (-1,0) 	: random.random(),
#         (1,0)	: random.random()
#         }.get)
#     print(f"y equals after {(time.time()-t0):.2f}s")

# def calcprimes(n):
#     t0 = time.time()
#     ps = 0
#     for i in range(n):
#         prime = True
#         for j in range(2,i-1):
#             if i % j == 0:
#                 prime = False 
#                 break 
#         if prime:
#             ps += 1
#     #printres("calcprimes",time.time()-t0)

# def runparallel(n):
#     t0 = time.time()
#     calls = []
#     for i in range(n):
#         calls.append(subprocess.Popen("python tester.py primes 20000",shell=True))
    
#     for p in calls:
#         p.wait()
#     printres(runparallel,time.time()-t0)


# def testparallel(n):
#     res = [] 

#     for i in range(1,n+1):
#         print(f"starting {i}/{n}")
#         t0 = time.time() 
#         runparallel(i)
#         dt = time.time()-t0 

#         tpc = i * 20000 / dt  
#         res.append(tpc)

#     plt.plot(res)
#     plt.show()


# funcs = {
#     "primes" : calcprimes,
#     "parallel" : runparallel,
#     "tp" : testparallel
# }
    

def mul(a,b):
    return a * b 
 
if __name__ == "__main__" and False :
    t0 = time.time() 
    for i in range(100000000000000000000,500000000000000000000,100000000000000):
        x = mul(i,10)
    print(f"func took {(time.time()-t0):.2f}")
    
    t0 = time.time() 
    for i in range(100000000000000000000,500000000000000000000,100000000000000):
        if i < 2000000:
            x = i * 10
    print(f"func took {(time.time()-t0):.2f}")
if __name__ == "__main__" and False :
    import numpy 
    import random   
    l = 64
    w = 13
    channels = 9
    steps_per_snake = 5
    #Python arrays
    t0 = time.time()

    arr_sizes = range(2,20,2)
    simul = 32

    t_new = [] 
    t_keep = []
    for arr_size in arr_sizes:

        #Test n_games iters 
        t0 = time.time() 
        t_tot = 0 
        
        for smooth in range(5):
            for game in range(simul):

                    snake = numpy.zeros(shape=(channels,arr_size,arr_size))
                    for episode_step in range(1000):

                        for game_i in range(simul):

                            for snake_bit in range(steps_per_snake):
                                y,z = random.randint(0,arr_size-1),random.randint(0,arr_size-1)
                                snake[0,y,z] = 1 
                        
                        #Transfer history
                        snake = numpy.roll(snake,1,axis=0)
                        snake[0] = numpy.zeros(shape=(arr_size,arr_size))

                                            
            t_tot += time.time()-t0
        t_new.append(t_tot/5)

        #Test 1 iter making n_games matrices
        t0 = time.time() 
        t_tot = 0 

        for smooth in range(5):

            snake = numpy.zeros(shape=(simul,channels,arr_size,arr_size))
            for episode_step in range(1000):

                for game_i in range(simul):

                    for snake_bit in range(steps_per_snake):
                        y,z = random.randint(0,arr_size-1),random.randint(0,arr_size-1)
                        #Only change the first channel
                        snake[game_i,0,y,z] = 1 
                
                #Transfer history
                snake = numpy.roll(snake,1,axis=1)
                snake[:,0] = numpy.zeros(shape=(simul,arr_size,arr_size))
                

                
            t_tot += time.time()-t0
        t_keep.append(t_tot/5)

    plt.scatter(arr_sizes,t_new,c="red",label="create new arrs")
    plt.scatter(arr_sizes,t_keep,c="cyan",label="keep old arrs")
    plt.legend()
    plt.show()


if __name__ == "__main__" and True :
    import numpy 
    import random   
    l = 64
    w = 13
    channels = 12
    steps_per_snake = 30
    #Python arrays
    t0 = time.time()

    arr_sizes = range(10,20,1)
    plays = 35
    simul = 16
    smoother = 25
    t_p_new = []
    t_n_new = [] 
    t_p_keep = []
    t_n_keep = []
    for arr_size in arr_sizes:

        #Test n_games iters 
        t0 = time.time() 
        t_tot = 0 
        
        for smooth in range(smoother):
            for game in range(simul):
                    snake = [[[0 for xx in range(arr_size)] for yy in range(arr_size)] for cc in range(channels)]
                    for episode_step in range(plays):

                        for snake_bit in range(steps_per_snake):
                            y,z = random.randint(0,arr_size-1),random.randint(0,arr_size-1)
                            snake[0][y][z] = 1 
                        
                        #Transfer history
                        snake = [[[0 for xx in range(arr_size)] for yy in range(arr_size)]] + snake[1:]

                                            
            t_tot += time.time()-t0
        t_p_new.append(t_tot/5)

        #Test 1 iter making n_games matrices
        t0 = time.time() 
        t_tot = 0 

        for smooth in range(smoother):

            snake = [[[[0 for xx in range(arr_size)] for yy in range(arr_size)] for cc in range(channels)] for gg in range(simul)]
            for episode_step in range(plays):

                for game_i in range(simul):

                    for snake_bit in range(steps_per_snake):
                        y,z = random.randint(0,arr_size-1),random.randint(0,arr_size-1)
                        #Only change the first channel
                        snake[game_i][0][y][z] = 1 
                
                #Transfer history
                snake = [[[[0 for xx in range(arr_size)] for yy in range(arr_size)]] + snake[gg][1:] for gg in range(simul)]
                

                
            t_tot += time.time()-t0
        t_p_keep.append(t_tot/5)

    for arr_size in arr_sizes:

        #Test n_games iters 
        t0 = time.time() 
        t_tot = 0 
        
        for smooth in range(smoother):
            for game in range(simul):

                    snake = numpy.zeros(shape=(channels,arr_size,arr_size))
                    for episode_step in range(plays):

                        for snake_bit in range(steps_per_snake):
                            y,z = random.randint(0,arr_size-1),random.randint(0,arr_size-1)
                            snake[0,y,z] = 1 
                    
                        #Transfer history
                        snake = numpy.roll(snake,1,axis=0)
                        snake[0] = numpy.zeros(shape=(arr_size,arr_size))

                                            
            t_tot += time.time()-t0
        t_n_new.append(t_tot/5)

        #Test 1 iter making n_games matrices
        t0 = time.time() 
        t_tot = 0 

        for smooth in range(smoother):

            snake = numpy.zeros(shape=(simul,channels,arr_size,arr_size))
            for episode_step in range(plays):

                for game_i in range(simul):

                    for snake_bit in range(steps_per_snake):
                        y,z = random.randint(0,arr_size-1),random.randint(0,arr_size-1)
                        #Only change the first channel
                        snake[game_i,0,y,z] = 1 
                
                #Transfer history
                snake = numpy.roll(snake,1,axis=1)
                snake[:,0] = numpy.zeros(shape=(simul,arr_size,arr_size))
                

                
            t_tot += time.time()-t0
        t_n_keep.append(t_tot/5)
    plt.scatter(arr_sizes,t_p_new,c="red",label="python new arrs")
    plt.scatter(arr_sizes,t_p_keep,c="orange",label="python old arrs")
    plt.scatter(arr_sizes,t_n_new,c="blue",label="numpy new arrs")
    plt.scatter(arr_sizes,t_n_keep,c="cyan",label="numpy old arrs")
    plt.legend()
    plt.show()