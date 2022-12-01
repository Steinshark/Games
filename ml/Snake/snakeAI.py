from cProfile import label
from filecmp import clear_cache
from cgi import print_exception
from operator import le
from turtle import back, shape
import pygame
from random import randint, sample
import random
import time
import pprint
import networks
import json
import numpy as np
import os
import torch
import torch.nn as nn
from multiprocessing import Pool, Process
from matplotlib import pyplot as plt
import traceback
import tkinter as tk
from tkinter import scrolledtext as tk_st

class SnakeGame:

	def __init__(self,w,h,fps=30,device=torch.device('cpu'),encoding_type="CNN"):
		self.width = w
		self.height = h

		self.food = (randint(0,self.width - 1),randint(0,self.height - 1))
		self.prev_food = self.food
		while self.food == (0,0):
				self.food = (randint(0,self.width - 1),randint(0,self.height - 1))

		self.snake = [[randint(0,self.width - 1),randint(0,self.height - 1)]]
		self.prev_snake = self.snake
		self.colors = {"FOOD" : (255,20,20),"SNAKE" : (20,255,20)}

		self.frame_time = 1 / fps

		self.snapshot_vector = [[0 for x in range(self.height)] for i in range(self.width)]

		self.direction = (1,0)
		self.device = device
		self.data = []
		self.encoding_type=encoding_type

	def play_game(self,window_x,window_y,training_match=True,model=None):


		if not window_x == window_y:
			print(f"invalid game_size {window_x},{window_y}.\nDimensions must be equal")
			return

		square_width 	= window_x / self.width
		square_height 	= window_y / self.height


		#Display setup

		pygame.init()
		self.window = pygame.display.set_mode((window_x,window_y))
		pygame.display.set_caption("AI Training!")


		self.output_vector = [0,0,0,1]
		game_running = True

		while game_running:

			#reset window and get events
			self.window.fill((0,0,0))
			pygame.event.pump()
			t_start = time.time()
			keys = pygame.key.get_pressed()
			f_time = t_start - time.time()
			#Draw snake and food
			if training_match:
				self.update_movement()
				self.create_input_vector()

			else:
				assert model is not None

				#Get the move value estimates
				y_feed = torch.tensor(self.game_to_model(self.create_input_vector()),dtype=torch.float)
				model_out = model.forward(y_feed)

				#Find "best" move
				w,s,a,d = model_out.cpu().detach().numpy()
				print([w,a,s,d])
				#Check for manual override
				keys= pygame.key.get_pressed()
				if True in [keys[pygame.K_w],keys[pygame.K_a],keys[pygame.K_s],keys[pygame.K_d]]:
					print("overriding ML")
					self.update_movement(player_input=True)
				else:
					self.update_movement(player_input=False,w=w,s=s,a=a,d=d)


			#Draw the Snake
			for coord in self.snake:
				x,y = coord[0] * square_width,coord[1] * square_height
				new_rect = pygame.draw.rect(self.window,self.colors["SNAKE"],pygame.Rect(x,y,square_width,square_height))
			#Draw the food
			x,y = self.food[0] * square_width,self.food[1] * square_height
			food_rect = pygame.draw.rect(self.window,self.colors["FOOD"],pygame.Rect(x,y,square_width,square_height))
			#Update display
			pygame.display.update()


			#Movement
			next_x = self.snake[0][0] + self.direction[0]
			next_y = self.snake[0][1] + self.direction[1]
			#Check for collision wtih wall
			if next_x >= self.width or next_y >= self.height or next_x < 0 or next_y < 0:
				game_running = False
			next_head = (next_x , next_y)
			#Check for collision with self
			if next_head in self.snake:
				print("you lose!")
				game_running = False
			#Check if snake ate food
			if next_head == self.food:
				self.food = (randint(0,self.width - 1),randint(0,self.height - 1))
				self.snake = [next_head] + self.snake
			#Normal Case
			else:
				self.snake = [next_head] + self.snake[:-1]
			#Check for vector request
			if keys[pygame.K_p]:
				print(f"input vect: {self.vector}")
				print(f"\n\noutput vect:{self.output_vector}")
			#Keep constant frametime
			self.data.append({"x":self.input_vector,"y":self.output_vector})
			if self.frame_time > f_time:
				time.sleep(self.frame_time - f_time)


		self.save_data()

	def save_data(self):
		x = []
		y = []
		for item in self.data[:-1]:
			x_item = np.ndarray.flatten(np.array(item["x"]))
			y_item = np.array(item["y"])

			x.append(x_item)
			y.append(y_item)

		x_item_final = np.ndarray.flatten(np.array(self.data[-1]["x"]))
		y_item_final = list(map(lambda x : x * -1,self.data[-1]["y"]))

		x.append(x_item_final)
		y.append(y_item_final)

		x = np.array(x)
		y = np.array(y)

		if not os.path.isdir("experiences"):
			os.mkdir("experiences")

		i = 0
		fname = f"exp_x_{i}.npy"
		while os.path.exists(os.path.join("experiences",fname)):
			i += 1
			fname = f"exp_x_{i}.npy"
		np.save(os.path.join("experiences",fname),x)
		np.save(os.path.join("experiences",f"exp_y_{i}.npy"),y)

	def game_to_model(self,x):
		return np.ndarray.flatten(np.array(x))

	def update_movement(self,player_input=False,w=0,s=0,a=0,d=0):

		if player_input:
			pygame.event.pump()
			keys = pygame.key.get_pressed()
			w,s,a,d = (0,0,0,0)

			if keys[pygame.K_w]:
				w = 1
			elif keys[pygame.K_s]:
				s = 1
			elif keys[pygame.K_a]:
				a = 1
			elif keys[pygame.K_d]:
				d = 1
			else:
				return
			self.output_vector = [w,s,a,d]

		self.movement_choices = {
			(0,-1) 	: w,
			(0,1) 	: s,
			(-1,0) 	: a,
			(1,0)	: d}

		self.direction = max(self.movement_choices,key=self.movement_choices.get)

	def train_on_game(self,model,visible=True,epsilon=.2,bad_opps=True):
		window_x, window_y = (1300,900)
		experiences = []
		rewards = {"die":-1,"food":1,"idle":-.1}
		score = 0
		
		#setup
		assert model is not None
		square_width 	= window_x / self.width
		square_height 	= window_y / self.height
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
				input_vector = torch.reshape(input_vector,(1,6,self.height,self.width))
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
			if next_head[0] >= self.width or next_head[1] >= self.height or next_head[0] < 0 or next_head[1] < 0 or next_head in self.snake or (bad_opps and (old_dir[0]*-1,old_dir[1]*-1) == self.direction):
				experiences.append({'s':input_vector,'r':rewards['die'],'a':self.direction,'s`':'terminal'})
				return experiences, score,lived

			#Check eat food
			if next_head == self.food:
				eaten_since = 0
				self.snake = [next_head] + self.snake
				self.food = (randint(0,self.width - 1),randint(0,self.height - 1))
				while self.food in self.snake:
					self.food = (randint(0,self.width - 1),randint(0,self.height - 1))

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
			if eaten_since > self.width*self.height*2:
				reward = rewards['idle']
				experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`':self.get_state_vector()})
				return experiences, score, lived
		return experiences, score, lived

	def get_state_vector(self):

		if self.encoding_type == "old":
			#Build x by y vector for snake
			input_vector = [[0 for x in range(self.height)] for y in range(self.width)]

			#Head of snake == 1
			input_vector[self.snake[0][1]][self.snake[0][0]] = 1

			#Rest of snake == -1
			for piece in self.snake[1:]:
				input_vector[piece[1]][piece[0]] = -1

			#Build x by y vector for food placement
			food_placement = [[0 for x in range(self.height)] for y in range(self.width)]
			food_placement[self.food[1]][self.food[0]] = 1
			input_vector += food_placement

		elif self.encoding_type == "3_channel":
			enc_vectr = []
			flag= False
			enc_vectr = [[[0 for x in range(self.width)] for y in range(self.height)] for _ in range(3)]
			enc_vectr[0][self.snake[0][1]][self.snake[0][0]] = 1

			for pos in self.snake[1:]:
				x,y = pos
				enc_vectr[1][y][x] = 1
			enc_vectr[2][self.food[1]][self.food[0]] = 1
			#for x,y in [(i%self.width,int(i/self.height)) for i in range(self.width*self.height)]:
			#	enc_vectr += [int((x,y) == self.snake[0]), int((x,y) in self.snake[1:]),int((x,y) == self.food)]
			enc_vectr = torch.reshape(torch.tensor(np.array(enc_vectr),dtype=torch.float,device=self.device),(3,self.width,self.height))
			#input(xcl[0].shape)
			return enc_vectr

		elif self.encoding_type == "6_channel":
			enc_vectr = torch.zeros((6,self.height,self.width))
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

			ret =  torch.reshape(enc_vectr,(6,self.height,self.width))
			return ret

		elif self.encoding_type == "one_hot":
			#Build x by y vector for snake
			snake_body = [[0 for x in range(self.width)] for y in range(self.height)]
			snake_head = [[0 for x in range(self.width)] for y in range(self.height)]

			#Head of snake == 1
			snake_head[self.snake[0][1]][self.snake[0][0]] = 1

			#Rest of snake
			for piece in self.snake[1:]:
				snake_body[piece[1]][piece[0]] = 1

			#Food
			food_placement = [[0 for x in range(self.width)] for y in range(self.height)]
			food_placement[self.food[1]][self.food[0]] = 1

			input_vector = snake_head + snake_body + food_placement
		#Translate to numpy and flatten
		np_array = np.ndarray.flatten(np.array(input_vector))

		#Translate to tensor
		tensr = torch.tensor(np_array,dtype=torch.float,device=self.device)
		return torch.tensor(np_array,dtype=torch.float,device=self.device)


class Trainer:

	def __init__(self,game_w,game_h,visible=True,loading=True,PATH="models",fps=200,loss_fn=torch.optim.Adam,optimizer_fn=nn.MSELoss,lr=1e-6,wd=1e-6,name="generic",gamma=.98,architecture=[256,32],gpu_acceleration=False,epsilon=.2,m_type="FCN"):
		self.PATH = PATH
		self.fname = name
		self.m_type = m_type
		self.input_dim = game_w * game_h * 3

		if m_type == "FCN":
			self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture)
			self.learning_model = networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture)
			self.encoding_type = "one_hot"

		elif m_type == "CNN":
			self.input_shape = (1,3,game_w,game_h)
			self.target_model 	= networks.ConvolutionalNetwork(channels=3,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture,input_shape=self.input_shape)
			self.learning_model = networks.ConvolutionalNetwork(channels=3,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture,input_shape=self.input_shape)
			self.encoding_type = "6_channel"

		self.w = game_w
		self.h = game_h
		self.gpu_acceleration = gpu_acceleration

		if gpu_acceleration:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.target_model.to(self.device)
		self.learning_model.to(self.device)

		self.visible = visible
		self.movement_repr_tuples = [(0,-1),(0,1),(-1,0),(1,0)]
		self.gamma = gamma
		self.fps = fps
		self.loss_fn = loss_fn
		self.optimizer_fn = optimizer_fn
		self.lr = lr
		self.wd = wd
		self.epsilon = epsilon
		self.e_0 = self.epsilon
		self.architecture = architecture
		settings = {
			"arch" : architecture,
			"loss_fn" : self.loss_fn,
			"optim_fn": self.optimizer_fn,
			"lr"		: self.lr,
			"wd"		: self.wd,
			"epsilon"	: self.epsilon,
			"y"			: self.gamma
		}
		import pprint

	def train(self,episodes=1000,train_every=1000,replay_buffer=32768,sample_size=128,batch_size=32,epochs=10,early_stopping=True,transfer_models_every=2000,verbose=True,iters=3):
		scored = [] 
		lived = [] 
		
		for i in range(iters):
			self.high_score = 0
			self.best = 0
			clear_every = 2
			experiences = []
			replay_buffer_size = replay_buffer
			t0 = time.time()
			high_scores = []
			trained = False

			scores = []
			lives = []

			picking = False
			for e_i in range(int(episodes)):
				#Play a game and collect the experiences
				game = SnakeGame(self.w,self.h,fps=100000,encoding_type=self.encoding_type,device=self.device)
				exp, score,lived_for = game.train_on_game(self.learning_model,visible=self.visible,epsilon=self.epsilon)

				scores.append(score+1)
				lives.append(lived_for)

				if score > self.high_score:
					self.high_score = score
				experiences += exp

				if len(experiences) > replay_buffer:
					experiences = experiences[int(-.8*replay_buffer):]

				#If training on this episode
				if e_i % train_every == 0 and not e_i == 0 and not len(experiences) <= sample_size:
					trained = True
					#Change epsilon within window of .1 to .4
					if (e_i/episodes) > .1 and self.epsilon > .01:
						e_range_percent_complete = ((e_i/episodes) - .1) / .4
						self.epsilon = self.e_0 - (self.e_0 * e_range_percent_complete)

					if verbose and e_i % 1024 == 0:
						print(f"[Episode {str(e_i).rjust(len(str(episodes)))}/{int(episodes)}  -  {(100*e_i/episodes):.2f}% complete\t{(time.time()-t0):.2f}s\te: {self.epsilon:.2f}\thigh_score: {self.high_score}] lived_avg: {sum(lives[-1000:])/len(lives[-1000:]):.2f} score_avg: {sum(scores[-1000:])/len(scores[-1000:]):.2f}")
					t0 = time.time()

					#Check score
					if self.high_score > self.best:
						self.best = self.high_score
					high_scores.append(self.high_score)
					self.high_score = 0


					best_sample = []

					if picking:
						blacklist = []
						indices = [i for i, item in enumerate(experiences) if not item['r'] < 0]
						quality = 100 * len(indices) / sample_size

						if verbose and e_i % 1000 == 0:
							print(f"quality of exps is {(100*quality / len(indices)):.2f}%")
						while not len(best_sample) == sample_size:
							if random.uniform(0,1) < .5:
								if len(indices) > 0:
									i = indices.pop(0)
									blacklist.append(i)
									best_sample.append(experiences[i])
								else:
									rand_i = random.randint(0,len(experiences)-1)
									while  rand_i in blacklist:
										rand_i = random.randint(0,len(experiences)-1)
									best_sample.append(experiences[rand_i])
							else:
									rand_i = random.randint(0,len(experiences)-1)
									while  rand_i in blacklist:
										rand_i = random.randint(0,len(experiences)-1)
									best_sample.append(experiences[rand_i])
						if verbose and e_i % 1000 == 0:
							quality = sum(map(lambda x : int(not x['r'] in [0]),best_sample))
							print(f"quality score {(100*quality/len(best_sample)):.2f}%")
					else:
						best_sample = random.sample(experiences,sample_size)


					#Train
					self.train_on_experiences(best_sample,batch_size=batch_size,epochs=epochs,early_stopping=early_stopping,verbose=e_i % 1024 == 0)
				if (e_i % transfer_models_every) == 0 and not e_i == 0 and trained:
					self.transfer_models(transfer=True,verbose=verbose)


			#Take score and lived data and shorten it
			smooth = int(episodes / 100)
			scores = [sum(scores[i:i+smooth])/smooth for i in range(0,int(len(scores)),smooth)]
			lives = [sum(lives[i:i+smooth])/smooth for i in range(0,int(len(lives)),smooth)]

			if len(lived) == 0:
				scored = scores
				lived = lives 
			else:
				scored = [scored[i] + scores[i] for i in range(len(scored))] 
				scored = [lived[i] + lives[i] for i in range(len(lived))] 
		
		lived = [l/iters for l in lived]
		scored = [s/iters for s in scored]

		#Save a fig for results

		#Top plot for avg. score, bottom plot for avg. time lived
		fig, axs = plt.subplots(2,1)
		fig.set_size_inches(19.2,10.8)

		#Plot data
		axs[0].plot([i*smooth for i in range(len(scores))],scored,label="scores",color='green')
		axs[1].plot([i*smooth for i in range(len(lived))],lived,label="lived for",color='cyan')
		axs[0].legend()
		axs[1].legend()
		axs[0].set_title(f"{self.architecture}-{str(self.loss_fn).split('.')[-1][:-2]}-{str(self.optimizer_fn).split('.')[-1][:-2]}-ep{epochs}-lr{self.lr}-bs{batch_size}-te{train_every}-rb{replay_buffer}-ss{sample_size}")

		#Save fig to figs directory
		if not os.path.isdir("figs"):
			os.mkdir("figs")
		fig.savefig(os.path.join("figs",f"{self.architecture}-{str(self.loss_fn).split('.')[-1][:-2]}-{str(self.optimizer_fn).split('.')[-1][:-2]}-ep{epochs}-lr{self.lr}-bs{batch_size}-te{train_every}-rb{replay_buffer}-ss{sample_size}.png"),dpi=100)


		#Return the best score, high scores of all episode blocks, scores, and steps lived
		return scores,lived

	def train_on_experiences(self,big_set,epochs=100,batch_size=8,early_stopping=True,verbose=False):
		for epoch_i in range(epochs):
			t0 = time.time()
			#Printing things
			if verbose and print(f"EPOCH {epoch_i}:\n\t",end='training['): pass
			next_percent = .02

			#Batch the sample set
			batches = [[big_set[i * n] for n in range(batch_size)] for i in range(int(len(big_set)/batch_size))]

			#Measrure losses and prepare for early stopping
			c_loss = 0
			prev_loss = 999999999999999

			#For each batch
			for i,batch in enumerate(batches):

				#Get a list (tensor) of all initial game states
				initial_states = torch.stack(([torch.reshape(exp['s'],(6,self.h,self.w)) for exp in batch]))
				#input(f"shape of one is {initial_states[0].shape}")
				#Make predictions of current states
				predictions = self.learning_model(initial_states)
				#Print progress of epoch
				if verbose:
					while (i / len(batches)) > next_percent:
						print("=",end='',flush=True)
						next_percent += .02

				#Get chosen action from the experience set e {0,0,0,0}
				chosen_action = [self.movement_repr_tuples.index(exp['a']) for exp in batch]

				# prepare for the adjusted values
				vals_target_adjusted = torch.zeros((batch_size,4))
				#input(f"init vta with size {vals_target_adjusted}")

				#Apply Bellman
				for index,action in enumerate(chosen_action):

					# If state was terminal, use target reward
					if batch[index]['s`'] == 'terminal' or batch[index]['r'] > 0:
						target = batch[index]['r']
					# If not terminal, use Bellman Equation
					else:
						next_state_val = torch.max(self.target_model(torch.reshape(batch[index]['s`'],(1,6,self.h,self.w))))
						target = batch[index]['r'] + (self.gamma * next_state_val)

					#Update with corrected value
					vals_target_adjusted[index,action] = target
					#input(f"target adj is now {vals_target_adjusted}")

				#Calculate error
				for param in self.learning_model.parameters():
					param.grad = None

				loss = self.learning_model.loss(vals_target_adjusted,predictions)
				c_loss += loss

				#Perform grad descent
				loss.backward()
				self.learning_model.optimizer.step()

			if early_stopping and c_loss > prev_loss:
				if verbose and print(f"] - early stopped on {epoch_i} at loss={c_loss} in {(time.time()-t0):.2f}s"): pass
				break
			prev_loss = c_loss
			if verbose and print(f"] loss: {c_loss:.4f} in {(time.time()-t0):.2f}"): pass
		if verbose:
			print("\n\n\n")

	def transfer_models(self,transfer=False,verbose=False):
		if transfer:
			if verbose:
				print("\ntransferring models\n\n")
			#Save the models

			torch.save(self.learning_model.state_dict(),os.path.join(self.PATH,f"{self.fname}_lm_state_dict"))
			#Load the learning model as the target model
			if self.m_type == "FCN":
				self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,lr=self.lr,wd=self.wd,architecture=self.architecture)
			elif self.m_type == "CNN":
				self.target_model = networks.ConvolutionalNetwork(channels=3,loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,lr=self.lr,wd=self.wd,architecture=self.architecture,input_shape=self.input_shape)

			self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))
			self.target_model.to(self.device)


def run_iteration(name,width,height,visible,loading,path,architecture,loss_fn,optimizer_fn,lr,wd,epsilon,epochs,episodes,train_every,replay_buffer,sample_size,batch_size,gamma,early_stopping,model_type):
	try:
		t1 = time.time()
		print(f"starting process {name}")
		trainer = Trainer(width,height,visible=visible,loading=loading,PATH=path,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,name=name,gamma=gamma,architecture=architecture,epsilon=epsilon,m_type=model_type)
		avg_scores,lived = trainer.train(episodes=episodes,train_every=train_every,replay_buffer=replay_buffer,sample_size=sample_size,batch_size=batch_size,epochs=epochs,early_stopping=early_stopping,verbose=True)
		print(f"\t{name} scored {best_score} in {(time.time()-t1):.2f}s")
	except Exception as e:
		traceback.print_exception(e)
 
	return {"time":time.time()-t1,"loss_fn":str(loss_fn).split(".")[-1].split("'")[0],"optimizer_fn":str(optimizer_fn).split(".")[-1].split("'")[0],"lr":lr,"wd":wd,"epsilon":epsilon,"epochs":epochs,"episodes":episodes,"train_every":train_every,"replay_buffer":replay_buffer,"sample_size":sample_size, "batch_size":batch_size,"gamma":gamma,"architecture":architecture,"best_score":best_score,"all_scores":all_scores,"avg_scores":avg_scores,"lived":lived}


if __name__ == "__main__" and True :
	for dir in ["models","sessions","figs"]:
		if not os.path.isdir(dir):
			os.mkdir(dir)

	lived = [] 
	scored = []

	#Long Model, quick Training 
	#trainer = Trainer(17,14,visible=False,loading=False,PATH="models",architecture=[[6,16,5],[16,16,5],[16,4,3],[1216,32],[32,4]],loss_fn=torch.nn.HuberLoss ,optimizer_fn=torch.optim.Adam,lr=.001,wd=0,name="LongModelQ",gamma=.97,epsilon=.4,m_type="CNN",gpu_acceleration=False)
	#l = trainer.train(episodes=100e4 ,train_every=16,replay_buffer=1024,sample_size=32,batch_size=16,epochs=1,transfer_models_every=128)
	
	#Short Model, quick Training 
	#trainer = Trainer(17,14,visible=False,loading=False,PATH="models",architecture=[[6,16,5],[16,8,5],[1904,4]],loss_fn=torch.nn.HuberLoss ,optimizer_fn=torch.optim.Adam,lr=.001,wd=0,name="ShortModelQ",gamma=.97,epsilon=.4,m_type="CNN",gpu_acceleration=False)
	#l = trainer.train(episodes=100e4 ,train_every=16,replay_buffer=1024,sample_size=32,batch_size=16,epochs=1,transfer_models_every=128)

	#Short Model, long Training 
	#trainer = Trainer(17,14,visible=False,loading=False,PATH="models",architecture=[[6,16,5],[16,8,5],[1904,4]],loss_fn=torch.nn.HuberLoss ,optimizer_fn=torch.optim.Adam,lr=.0001,wd=0,name="ShortModelL",gamma=.97,epsilon=.4,m_type="CNN",gpu_acceleration=False)
	#l = trainer.train(episodes=2.5e4 ,train_every=512,replay_buffer=1024*8,sample_size=1024,batch_size=16,epochs=1,transfer_models_every=1024)
	#scores, lives = l[0],l[1]
	#import json
	#import sys
	#fname = os.path.join("sessions","saved_states2.txt")
	#if len(sys.argv) > 1:
	#	fname = os.path.join("sessions",sys.argv[1])
	#with open(fname,"w") as file:
	#	file.write(json.dumps(l))
	#exit()
	loss_fns = [torch.nn.HuberLoss,torch.nn.MSELoss]#,torch.nn.L1Loss]
	optimizers = [torch.optim.Adam]

	learning_rates = [1e-3,1e-4,1e-5,1e-6]
	episodes = 1e6

	gamma = [.97]
	epsilon=[.4]
	train_every = [32,128,1024]
	replay_buffer =[512,1024,4096,16384]
	sample_size = [128,512,1024,2048]
	batch_sizes = [1,8,16,32]#2,16,32,64]#,4,32]
	epochs = [1,2,5]
	w_d = [0,1e-3,1e-6]
	architectures = [[[6,32,5],[7072,64],[64,4]],[[6,16,5],[16,16,5],[16,8,3],[2280,4]],[[6,64,3],[64,16,5],[4560,4]]]
	i = 0
	args = []
	processes = []

	for l in loss_fns:
		for o in optimizers:
				for y in gamma:
					for e in epochs:
						for lr in learning_rates:
							for t in train_every:
								for r in replay_buffer:
									for s in sample_size:
										for b in batch_sizes:
											for a in architectures:
												for h in epsilon:
													for w in w_d:
														if r < s or r < b or s < b:
															pass
														else:
															args.append((i,17,13,False,False,"models",a,l,o,lr,w,h,e,episodes,t,r,s,b,y,True,"CNN"))
															i += 1

	if not input(f"testing {len(args)} trials, est. completion in {(.396 * (len(args)*episodes / 40)):.1f}s [{(.396*(1/3600)*(len(args)*episodes / 40)):.2f}hrs]. Proceed? [y/n] ") in ["Y","y","Yes","yes","YES"]: exit()

	random.shuffle(args)
	with Pool(4) as p:
		try:
			t0 = time.time()
			results = p.starmap(run_iteration,args)
			import json
			import sys
			fname = os.path.join("sessions","saved_states.txt")
			if len(sys.argv) > 1:
				fname = os.path.join("sessions",sys.argv[1])
			with open(fname,"w") as file:
				file.write(json.dumps(results))
			print(f"ran in {(time.time()-t0):.2f}s")
		except Exception as e:
			print("aborting")
			traceback.print_exception(e)
