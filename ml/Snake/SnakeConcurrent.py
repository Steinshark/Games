import pygame 

from random import randint 
import torch
import time 
import numpy as np 


#This class interfaces only with NP 
class Snake:

	def __init__(self,w,h,n_games=32,fps=30,device=torch.device('cpu'),encoding_type="CNN"):

		#Set global Vars
		self.grid_w = w
		self.grid_h = h
		self.num_games = n_games


		#Set game vars 
		self.snakes = [[]] * n_games
		self.foods = [(randint(1,self.grid_w - 1),randint(1,self.grid_h - 1))] * n_games
		self.prev_foods = self.foods 

		self.snakes = [[randint(0,self.grid_w - 1),randint(0,self.grid_h - 1)]] * n_games
		self.prev_snake = self.snakes

		self.snapshot_vector = [[0 for x in range(self.grid_h)] for i in range(self.grid_w)] * n_games

		self.directions = [(1,0)] * n_games
		self.device = device
		self.data = []
		self.encoding_type=encoding_type

	def update_movements(self,model_outputs):
		
		for out in model_outputs:
			w,a,s,d = out
			movement_choices = {
				(0,-1) 	: w,
				(0,1) 	: s,
				(-1,0) 	: a,
				(1,0)	: d}

		self.direction = max(self.movement_choices,key=self.movement_choices.get)

	def train_on_game(self,model,visible=True,epsilon=.2,bad_opps=True):
		window_x, window_y = (1920,1080)
		experiences = []
		rewards = {"die":-1,"food":1,"idle":-.1}
		score = 0
		
		#setup
		assert model is not None
		square_width 	= window_x / self.grid_w
		square_height 	= window_y / self.grid_h
		game_running = True
		eaten_since = 0
		lived = 0
		self.prev_frame = self.get_state_vector()
		#Game display
		if visible:
			pygame.init()
			self.window = pygame.display.set_mode((window_x,window_y))
			pygame.display.set_caption("AI Training!")

		#Game Loop
		while game_running:
			lived += 1
			#Get init states
			input_vector = self.get_state_vector()
			old_dir = self.direction
			self.prev_snake = self.snake
			self.prev_food = self.food

			#Update move randomly
			if random.random() < epsilon:
				while self.direction == old_dir:
					x = random.randint(-1,1)
					y = int(x == 0) * random.sample([1,-1],1)[0]
					self.direction = (x,y)
			else:
				input_vector = torch.reshape(input_vector,(1,6,self.grid_h,self.grid_w))
				movement_values = model.forward(input_vector)
				try:
					w,s,a,d = movement_values.cpu().detach().numpy()
				except ValueError:
					w,s,a,d = movement_values[0].cpu().detach().numpy()
				self.update_movement(w=w,s=s,a=a,d=d)

			#Game display
			if visible:
				self.window.fill((0,0,0))
				for coord in self.snake:
					x,y = coord[0] * square_width,coord[1] * square_height
					new_rect = pygame.draw.rect(self.window,self.colors["SNAKE"],pygame.Rect(x,y,square_width,square_height))
				x,y = self.food[0] * square_width,self.food[1] * square_height
				food_rect = pygame.draw.rect(self.window,self.colors["FOOD"],pygame.Rect(x,y,square_width,square_height))
				pygame.display.update()

			#Find New Head
			next_x = self.snake[0][0] + self.direction[0]
			next_y = self.snake[0][1] + self.direction[1]
			next_head = (self.snake[0][0] + self.direction[0] , self.snake[0][1] + self.direction[1])

			#Check lose
			if next_head[0] >= self.grid_w or next_head[1] >= self.grid_h or next_head[0] < 0 or next_head[1] < 0 or next_head in self.snake or (bad_opps and (old_dir[0]*-1,old_dir[1]*-1) == self.direction):
				experiences.append({'s':input_vector,'r':rewards['die'],'a':self.direction,'s`':'terminal'})
				return experiences, score,lived

			#Check eat food
			if next_head == self.food:
				eaten_since = 0
				self.snake = [next_head] + self.snake
				self.food = (randint(0,self.grid_w - 1),randint(0,self.grid_h - 1))
				while self.food in self.snake:
					self.food = (randint(0,self.grid_w - 1),randint(0,self.grid_h - 1))

				reward = rewards['food']
				score += 1
			#Check No Outcome
			else:
				self.snake = [next_head] + self.snake[:-1]
				reward = rewards["idle"]

			#Add to experiences

			experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`': self.get_state_vector()})

			eaten_since += 1

			#Check if lived too long
			if eaten_since > self.grid_w*self.grid_h*2:
				reward = rewards['idle']
				experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`':self.get_state_vector()})
				return experiences, score, lived
		return experiences, score, lived

	def get_state_vector(self):

		if self.encoding_type == "old":
			#Build x by y vector for snake
			input_vector = [[0 for x in range(self.grid_h)] for y in range(self.grid_w)]

			#Head of snake == 1
			input_vector[self.snake[0][1]][self.snake[0][0]] = 1

			#Rest of snake == -1
			for piece in self.snake[1:]:
				input_vector[piece[1]][piece[0]] = -1

			#Build x by y vector for food placement
			food_placement = [[0 for x in range(self.grid_h)] for y in range(self.grid_w)]
			food_placement[self.food[1]][self.food[0]] = 1
			input_vector += food_placement

		elif self.encoding_type == "3_channel":
			enc_vectr = []
			flag= False
			enc_vectr = [[[0 for x in range(self.grid_w)] for y in range(self.grid_h)] for _ in range(3)]
			enc_vectr[0][self.snake[0][1]][self.snake[0][0]] = 1

			for pos in self.snake[1:]:
				x,y = pos
				enc_vectr[1][y][x] = 1
			enc_vectr[2][self.food[1]][self.food[0]] = 1
			#for x,y in [(i%self.grid_w,int(i/self.grid_h)) for i in range(self.grid_w*self.grid_h)]:
			#	enc_vectr += [int((x,y) == self.snake[0]), int((x,y) in self.snake[1:]),int((x,y) == self.food)]
			enc_vectr = torch.reshape(torch.tensor(np.array(enc_vectr),dtype=torch.float,device=self.device),(3,self.grid_w,self.grid_h))
			#input(xcl[0].shape)
			return enc_vectr

		elif self.encoding_type == "6_channel":
			enc_vectr = torch.zeros((6,self.grid_h,self.grid_w))
			flag= False

			#Old SNAKE
			#Place head (ch0)
			enc_vectr[0][self.prev_snake[0][1]][self.prev_snake[0][0]] = 1
			#Place body (ch1)
			for pos in self.prev_snake[1:]:
				x = pos[0]
				y = pos[1]
				enc_vectr[1][y][x] = 1
			#Place food (ch2)
			enc_vectr[2][self.prev_food[1]][self.prev_food[0]] = 1

			#Cur SNAKE
			#Place head (ch3)
			enc_vectr[3][self.snake[0][1]][self.snake[0][0]] = 1
			#Place body (ch4)
			for pos in self.snake[1:]:
				x = pos[0]
				y = pos[1]
				enc_vectr[4][y][x] = 1
			#Place food (ch5)
			enc_vectr[5][self.food[1]][self.food[0]] = 1

			ret =  torch.reshape(enc_vectr,(6,self.grid_h,self.grid_w))
			return ret

		elif self.encoding_type == "one_hot":
			#Build x by y vector for snake
			snake_body = [[0 for x in range(self.grid_w)] for y in range(self.grid_h)]
			snake_head = [[0 for x in range(self.grid_w)] for y in range(self.grid_h)]

			#Head of snake == 1
			snake_head[self.snake[0][1]][self.snake[0][0]] = 1

			#Rest of snake
			for piece in self.snake[1:]:
				snake_body[piece[1]][piece[0]] = 1

			#Food
			food_placement = [[0 for x in range(self.grid_w)] for y in range(self.grid_h)]
			food_placement[self.food[1]][self.food[0]] = 1

			input_vector = snake_head + snake_body + food_placement
		#Translate to numpy and flatten
		np_array = np.ndarray.flatten(np.array(input_vector))

		#Translate to tensor
		tensr = torch.tensor(np_array,dtype=torch.float,device=self.device)
		return torch.tensor(np_array,dtype=torch.float,device=self.device)

