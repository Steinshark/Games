import numpy 
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
import random 

from stable_baselines3 import dqn

class Snake(gym.Env):

    def __init__(self,width,height):
        super(Snake,self).__init__()
        
        self.width = width 
        self.height = height 
        
        self.action_space = spaces.Discrete(4)
        max = 255*numpy.ones((3,height,width))
        min = 0*numpy.ones((3,height,width))
        self.observation_space = spaces.Box(low=0,high=1,shape=(3,height,width),dtype=numpy.uint8)
        self.snake = [(0,0)]
        self.food = (random.randint(0,height-1),random.randint(0,width-1))

    def step(self,action):
        if action == 0:
            movement = (0,-1)
        elif action == 1:
            movement = (0,1)
        elif action == 2:
            movement = (1,0)
        elif action == 3:
            movement = (-1,0)

        next_x, next_y = self.snake[0][0] + movement[0], self.snake[0][1] + movement[1]

        #Dies 
        if next_x > self.width - 1 or next_x < 0 or next_x > self.height - 1 or next_y < 0 or (next_x,next_y) in self.snake:
            return self.encoding(), -1, True, {} 

        #Eats
        elif (next_x,next_y) == self.food:
            self.snake = [(next_x,next_y)] + self.snake 
            self.spawn_food()
            return self.encoding(),1,False,{}
        
        #Lives
        else:
            self.snake = [(next_x,next_y)] + self.snake[:-1]
            return self.encoding(),0,False,{}
    
    def reset(self):
        self.snake = [(0,0)]
        self.food = (random.randint(0,self.height-1),random.randint(0,self.width-1))
        return self.encoding()
    
    #Spawn a new food | food is not in the snake
    def spawn_food(self):   
        x = random.randint(0,self.width-1) 
        y = random.randint(0,self.height-1)   
        self.food = (x,y)

        while self.food in self.snake:
            x = random.randint(0,self.width-1) 
            y = random.randint(0,self.height-1)   
            self.food = (x,y)
    
    def encoding(self):

        enc_vect = numpy.zeros((3,self.height,self.width),dtype=numpy.uint8)
        
        #Encode head
        enc_vect[0][self.snake[0][1]][self.snake[0][0]] = 255
        #Encode Rest 
        for pos in self.snake[1:]:
            enc_vect[1][pos[1]][pos[0]] = 255
        #Encode Food  
        enc_vect[2][self.food[1]][self.food[0]] = 255

        return enc_vect
        
if __name__ == "__main__":
    env = Snake(8,8)
    check_env(env)

    model = dqn.DQN("CnnPolicy",env,verbose=True)
    model.learn(total_timesteps=10000,log_interval=10)

