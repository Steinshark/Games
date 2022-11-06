import random
from matplotlib.dates import epoch2num 
import numpy 
import pygame 
import time 
import networks 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import copy

s_w = 600 
s_h = 600


class Game:     
    def __init__(self,x,y):

        self.width          = x
        self.height         = y 
        self.game_state     = "Playing"
        self.reward         = 0
        self.snake          = [(random.randint(0,x-1),random.randint(0,y-1))]
        self.prev_snake     = self.snake
        
        self.gen_food()
        self.prev_food = self.food
        
        self.rewards = {
                "eat" : 1,
                "die" : -1,
                "live": -.1  
            }
        self.direction = (1,0)

    def gen_food(self):
        self.food = (random.randint(0,self.width-1),random.randint(0,self.height-1))
        while self.food in self.snake:
            self.food = (random.randint(0,self.width-1),random.randint(0,self.height-1))
        
    def step(self,direction):
        
        #Calc next movement 
        step_x = direction[0]
        step_y = direction[1]
        next_head = (self.snake[0][0] + step_x,self.snake[0][1] + step_y)
        
        #Save old state 
        self.prev_snake = self.snake
        self.prev_food = self.food
        #Check Food 
        if next_head == self.food:
            self.snake = [next_head] + self.snake 
            self.reward = self.rewards["eat"]
            self.gen_food()
            return
        
        #Check death 
        elif next_head[0] < 0 or next_head[0] >= self.width or next_head[1] < 0 or next_head[1] >= self.height or next_head in self.snake:
            self.game_state = "Lost"
            self.reward = self.rewards["die"]
            return False 

        #Check normal case 
        else:
            self.snake = [next_head] + self.snake[:-1]
            self.reward = self.rewards["live"]
            return True 

    def get_repr(self,encoding="6Channel"):
        
        head_x = self.snake[0][0]
        head_y = self.snake[0][1]

        food_x = self.food[0]
        food_y = self.food[1]

        #3 Channel encoding {[head],[body],[food]}
        if encoding == "3Channel":
            base_vectr = numpy.zeros((3,self.height,self.width))
            
            #Encode head 
            base_vectr[0][head_y][head_x] = 1

            #Encode body 
            for component in self.snake[1:]:
                comp_x = component[0]
                comp_y = component[1]
                base_vectr[1][comp_y][comp_x] = 1 
            
            #Encode food 
            base_vectr[2][food_y][food_x] = 1

            return base_vectr
        
        elif encoding == "6Channel":

            base_vectr = numpy.zeros((6,self.width,self.height))
            
            #Encode current snake 
            base_vectr[0][head_y][head_x] = 1

            # body 
            for component in self.snake[1:]:
                comp_x = component[0]
                comp_y = component[1]
                base_vectr[1][comp_y][comp_x] = 1 
            
            # food 
            base_vectr[2][food_y][food_x] = 1

            #Encode old state
            head_x = self.prev_snake[0][0]
            head_y = self.prev_snake[0][1]

            food_x = self.prev_food[0]
            food_y = self.prev_food[1]

            #Encode head 
            base_vectr[3][head_y][head_x] = 1

            #Encode body 
            for component in self.prev_snake[1:]:
                comp_x = component[0]
                comp_y = component[1]
                base_vectr[4][comp_y][comp_x] = 1 
            
            #Encode food 
            base_vectr[5][food_y][food_x] = 1

            return base_vectr

    def update_dir(self,mode="user_in",fps=15,model=None):
        if not fps is None: 
            t0 = time.time()
            while time.time() - t0 < 1/fps:
                pygame.event.pump()

        
        if mode == "pygame":
        
            keys = pygame.key.get_pressed()
            keys = {(0,-1) : keys[pygame.K_w],
                    (-1,0) : keys[pygame.K_a],
                    (0,1)  : keys[pygame.K_s],
                    (1,0)  : keys[pygame.K_d]}
            for k in keys:
                if keys[k]:
                    self.direction = k
                    return k
            return self.direction
        
        elif mode == "model":
            repr = self.get_repr(encoding="6Channel")
            repr = torch.tensor(repr,dtype=torch.float)
            outs = model.forward(repr)


            return {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[torch.argmax(outs).item()]
            


class Trainer:

    def __init__(self):
        self.target_model = networks.ConvolutionalNetwork(6,nn.HuberLoss,torch.optim.SGD,.005,0,[[6,16,3],[16,8,5],[72,4]],(1,6,5,5))
        self.training_model = networks.ConvolutionalNetwork(6,nn.HuberLoss,torch.optim.SGD,.005,0,[[6,16,3],[16,8,5],[72,4]],(1,6,5,5))

    def train(self,episodes=20000):
        
        exps = []

        for i in range(episodes):
            g = Game()
            exps += train_game(g,self.training_model)
            
def play_game(g,visible=(True,600,600)):
    exps = {}

    if visible:

        #Setup variables
        screen_w = visible[1]
        screen_h = visible[2]
        box_width = screen_w / g.width
        box_height = screen_h / g.height

        #Setup Display Env
        pygame.init()
        window = pygame.display.set_mode((screen_w,screen_h))

        #Run game 
        while g.game_state == "Playing":
            window.fill((0,0,0))

            d = g.update_dir(mode="model",fps=.2)
            
            g.step(d)

            for box in g.snake:
                pygame.draw.rect(window,(0,255,150),pygame.Rect(box[0]*box_width,box[1]*box_height,box_width,box_height))
            pygame.draw.rect(window,(255,50,50),pygame.Rect(g.food[0]*box_width,g.food[1]*box_height,box_width,box_height))

            pygame.display.flip()

            print(g.get_repr("6Channel"))

def train_game(g:Game,model:networks.ConvolutionalNetwork):

    pygame.init()
    window = pygame.display.set_mode((600,600))

    exps = []
    b_w = s_w / g.width
    b_h = s_h / g.height

    while g.game_state == "Playing":
        window.fill((0,0,0))

        exp = {"s":g.get_repr() ,"a":None,"r":None,"s`":None,"done":None}
        g.update_dir(mode="model",fps=1000,model=model)
        exp["done"] = not g.step(g.direction)
        exp["r"] = g.reward
        exp["a"] = g.direction 
        exp["s`"] = g.get_repr()    

        exps.append(exp)
        for box in g.snake:
            pygame.draw.rect(window,(0,255,60),pygame.Rect(box[0]*b_w,box[1]*b_w,b_w,b_h))
        pygame.draw.rect(window,(255,60,60),pygame.Rect(g.food[0]*b_w,g.food[1]*b_h,b_w,b_h))

        pygame.display.flip()
    return exps

def train_model(experiences,model,batch_size=16,pool_size=128):
    #Copy the list and shuffle it 
    r_exps = copy.copy(experiences)
    random.shuffle(r_exps)

    #Sample a random set and batch the data 
    sample_set = random.sample(experiences,pool_size)
    batches = [[sample_set[i+k*batch_size] for i in range(batch_size)] for k in range(int(pool_size/batch_size))]

    for batch in batches:
        #Grab states and forward pass through model  
        s_states = torch.tensor(numpy.array(([(b["s"]) for b in batch])),dtype=torch.float)
        s_prime_states =  torch.tensor(numpy.array(([(b["s`"]) for b in batch])),dtype=torch.float)
        actions = torch.tensor(numpy.array([b['a'] for b in batch]),dtype=torch.float)

        next_state_predictions = model.forward(s_prime_states)
        next_state_rewards = [torch.argmax(t) for t in next_state_predictions]

        
        
        #Bellman Eq + final state check 
        final_eval = torch.clone(next_moves)


        #Gradient descent 


        input(best_moves)

def run(episodes):
    #Create a network to handle 6 channel 

    model = networks.ConvolutionalNetwork(6,nn.HuberLoss,torch.optim.SGD,.005,0,[[6,16,3],[16,8,5],[72,4]],(1,6,5,5))
    train_every = 100
    experiences = []

    for i in range(episodes):
        g= Game(5,5)
        experiences += train_game(g,model)
        if i % train_every == 0 and not i == 0:
            train_model(experiences,model)

if __name__ == "__main__":
    run(1000)
        




