from cgi import print_exception
from filecmp import clear_cache
from operator import le
from turtle import back
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

class SnakeGame:

	def __init__(self,w,h,fps=30,device=torch.device('cpu')):
		self.width = w
		self.height = h

		self.food = (randint(0,self.width - 1),randint(0,self.height - 1))

		while self.food == (0,0):
				self.food = (randint(0,self.width - 1),randint(0,self.height - 1))

		self.snake = [[0,0]]

		self.colors = {"FOOD" : (255,20,20),"SNAKE" : (20,255,20)}

		self.frame_time = 1 / fps

		self.snapshot_vector = [[0 for x in range(self.height)] for i in range(self.width)]

		self.direction = (1,0)
		self.device = device
		self.data = []

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

	def create_input_vector(self):
		self.input_vector = [[0 for x in range(self.height)] for y in range(self.width)]
		self.input_vector[self.snake[0][1]][self.snake[0][0]] = 1
		for piece in self.snake[1:]:
			self.input_vector[piece[1]][piece[0]] = -1
		food_placement = [[0 for x in range(self.height)] for y in range(self.width)]
		food_placement[self.food[1]][self.food[0]] = 1
		self.input_vector += food_placement
		return self.input_vector

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

	def train_on_game(self,model,visible=True,epsilon=.2):
		window_x, window_y = (600,600)
		experiences = []
		rewards = {"die":-10,"food":10,"live":0,"idle":0}
		score = 0
		#setup
		assert model is not None
		square_width 	= window_x / self.width
		square_height 	= window_y / self.height
		game_running = True
		eaten_since = 0

		#Game display
		if visible:
			pygame.init()
			self.window = pygame.display.set_mode((window_x,window_y))
			pygame.display.set_caption("AI Training!")

		#Game Loop
		while game_running:
			input_vector = self.get_state_vector()

			#Find next update_movement
			if random.random() < epsilon:
				x = random.randint(-1,1)
				y = int(x == 0) * random.sample([1,-1],1)[0]
				self.direction = (x,y)
			else:
				movement_values = model.forward(input_vector.to(self.device))
				w,s,a,d = movement_values.cpu().detach().numpy()
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


			#Game Logic
			next_x = self.snake[0][0] + self.direction[0]
			next_y = self.snake[0][1] + self.direction[1]

			next_head = (self.snake[0][0] + self.direction[0] , self.snake[0][1] + self.direction[1])

			#Check lose
			if next_head[0] >= self.width or next_head[1] >= self.height or next_head[0] < 0 or next_head[1] < 0 or next_head in self.snake:
				experiences.append({'s':input_vector,'r':rewards['die'],'a':self.direction,'s`':'terminal'})
				return experiences, score

			#Check eat food
			if next_head == self.food:
				eaten_since = 0
				self.snake = [next_head] + self.snake
				self.food = (randint(0,self.width - 1),randint(0,self.height - 1))
				while self.food in self.snake:
					self.food = (randint(0,self.width - 1),randint(0,self.height - 1))

				reward = rewards['food']
				score += 1
			else:
				self.snake = [next_head] + self.snake[:-1]
				reward = rewards['live']

			#Add to experiences
			experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`': self.get_state_vector()})
			eaten_since += 1

			#Dont kill it since epsilon should take care of it
			if eaten_since > 60:
				reward = rewards['idle']
				experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`':self.get_state_vector()})
				return experiences, score
		return experiences, score


	def get_state_vector(self,encoding="one_hot"):

		if encoding == "old":
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

		elif encoding == "one_hot":
			#Build x by y vector for snake
			snake_body = [[0 for x in range(self.width)] for y in range(self.height)]
			snake_head = [[0 for x in range(self.width)] for y in range(self.height)]
			#Head of snake == 1
			snake_head[self.snake[0][1]][self.snake[0][0]] = 1

			#Rest of snake
			for piece in self.snake[1:]:
				snake_body[piece[1]][piece[0]] = 1

			#Build x by y vector for food placement
			food_placement = [[0 for x in range(self.height)] for y in range(self.width)]
			food_placement[self.food[1]][self.food[0]] = 1
			input_vector = snake_head + snake_body + food_placement
		#Translate to numpy and flatten
		np_array = np.ndarray.flatten(np.array(input_vector))
		#Translate to tensor
		return torch.tensor(np_array,dtype=torch.float,device=self.device)


class Trainer:

	def __init__(self,game_w,game_h,visible=True,loading=True,PATH="E:\code\Scratch\models",fps=200,loss_fn=torch.optim.Adam,optimizer_fn=nn.MSELoss,lr=1e-6,wd=1e-6,name="generic",gamma=.98,architecture=[256,32],gpu_acceleration=False,epsilon=.2):
		self.PATH = PATH
		self.fname = name

		self.input_dim = game_w * game_h * 3
		self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture)
		self.learning_model = networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture)
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

	def train(self,episodes=1000,train_every=1000,replay_buffer=32768,sample_size=128,batch_size=32,epochs=10,early_stopping=True,transfer_models_every=2,verbose=False):

		self.high_score = 0
		self.best = 0
		clear_every = 2
		experiences = []
		replay_buffer_size = replay_buffer
		t0 = time.time()
		high_scores = []

		for e_i in range(int(episodes)):

			#Play a game and collect the experiences
			game = SnakeGame(self.w,self.h,fps=100)
			exp, score = game.train_on_game(self.learning_model,visible=self.visible,epsilon=self.epsilon)
			if score > self.high_score:
				self.high_score = score
			experiences += exp
			if len(experiences) > replay_buffer:
				experiences = experiences[int(-.8*replay_buffer):]

			#If training on this episode
			if e_i % train_every == 0 and not e_i == 0 and not len(experiences) <= sample_size:

				#Change epsilon
				if (e_i/episodes) > .1:
					self.epsilon *= .8

				if verbose:
					print(f"[Episode {str(e_i).rjust(len(str(episodes)))}/{int(episodes)}  -  {(100*e_i/episodes):.2f}% complete\t{(time.time()-t0):.2f}s\te: {self.epsilon:.2f}\thigh_score: {self.high_score}]")
				t0 = time.time()

				#Check score
				if self.high_score > self.best:
					self.best = self.high_score
				high_scores.append(self.high_score)
				self.high_score = 0

				#Make better experience set
				best = random.sample(experiences,sample_size)
				best_size = sum(map(lambda x : int(not x['r'] == 0) + ((2*int(x['r']) == 10)), best))

				for _ in range(1000):
					training_set = random.sample(experiences,sample_size)
					efficiency = sum(map(lambda x : int(not x['r'] == 0) + ((2*int(x['r']) == 10)), training_set))
					if efficiency > best_size:
						best = training_set
						best_size = efficiency

				#print(f"efficiency score is {(100* best_size / len(best)):.3f}")

				#Train
				self.train_on_experiences(best,batch_size=batch_size,epochs=epochs,early_stopping=early_stopping)
			if e_i % transfer_models_every == 0:
				self.transfer_models(transfer=True)

		return self.best,high_scores

	def train_on_experiences(self,big_set,epochs=100,batch_size=8,early_stopping=True,verbose=False):
		for epoch_i in range(epochs):

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
				initial_states = torch.stack(([exp['s'] for exp in batch]))

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
				vals_target_adjusted = torch.clone(predictions)

				#Apply Bellman
				for index,action in enumerate(chosen_action):

					# If state was terminal, use target reward
					if batch[index]['s`'] == 'terminal':
						target = batch[index]['r']
					# If not terminal, use Bellman Equation
					else:
						next_state_val = torch.max(self.target_model(batch[index]['s`']))
						target = batch[index]['r'] + (self.gamma * next_state_val)

					#Update with corrected value
					vals_target_adjusted[index,action] = target

				#Calculate error
				for param in self.learning_model.parameters():
					param.grad = None
				loss = self.learning_model.loss(predictions,vals_target_adjusted)
				c_loss += loss

				#Perform grad descent
				loss.backward()
				self.learning_model.optimizer.step()

			if early_stopping and c_loss > prev_loss:
				if verbose and print(f"] - early stopped on {epoch_i} at loss={c_loss}"): pass
				break
			prev_loss = c_loss
			if verbose and print(f"] loss: {c_loss}"): pass
		if verbose:
			print("\n\n\n")

	def transfer_models(self,transfer=False):
		if transfer:
			#Save the models
			torch.save(self.learning_model.state_dict(),os.path.join(self.PATH,f"{self.fname}_lm_state_dict"))

			#Load the learning model as the target model
			self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,lr=self.lr,wd=self.wd,architecture=self.architecture)
			self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))


def run_iteration(name,width,height,visible,loading,path,architecture,loss_fn,optimizer_fn,lr,wd,epsilon,epochs,episodes,train_every,replay_buffer,sample_size,batch_size,gamma,early_stopping):
	try:
		t1 = time.time()
		print(f"starting process {name}")
		trainer = Trainer(width,height,visible=visible,loading=loading,PATH=path,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,name=name,gamma=gamma,architecture=architecture,epsilon=epsilon)
		best_score,all_scores = trainer.train(episodes=episodes,train_every=train_every,replay_buffer=replay_buffer,sample_size=sample_size,batch_size=batch_size,epochs=epochs,early_stopping=early_stopping)
		print(f"\t{name} scored {best_score} in {(time.time()-t1):.2f}s")
	except Exception as e:
		traceback.print_exception(e)

	return {"time":time.time()-t1,"loss_fn":str(loss_fn),"optimizer_fn":str(optimizer_fn),"lr":lr,"wd":wd,"epsilon":epsilon,"epochs":epochs,"episodes":episodes,"train_every":train_every,"replay_buffer":replay_buffer,"sample_size":sample_size, "batch_size":batch_size,"gamma":gamma,"architecture":architecture,"best_score":best_score,"all_scores":all_scores}



if __name__ == "__main__":

	#run_iteration("test",5,5,True,False,"models",[128,32],torch.nn.MSELoss,torch.optim.Adam,.0001,1e-6,.5,1,5e5,2048,32768/2,128,8,.985,True)
	#exit()
	loss_fns = [torch.nn.MSELoss]
	optimizers = [torch.optim.Adam]

	learning_rates = [1e-4]
	episodes = 1e6

	gamma = [.99]
	epsilon=[.5]
	train_every = [1000,10000]
	replay_buffer = [128,4096,16384*2]
	sample_size = [64,8192]
	batch_sizes = [2,32]
	epochs = [1,4]
	w_d = [0]
	architectures = [[256,32],[16,16,16,16]]
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
														if r < s or r < b or s < b or t < s:
															pass
														else:
															args.append((i,10,10,False,False,"models",a,l,o,lr,w,h,e,episodes,t,r,s,b,y,True,))
															i += 1

	if not input(f"testing {len(args)} trials, est. completion in {(.396 * (len(args)*episodes / 40)):.1f}s [{(.396*(1/3600)*(len(args)*episodes / 40)):.2f}hrs]. Proceed? [y/n] ") in ["Y","y","Yes","yes","YES"]: exit()

	random.shuffle(args)

	with Pool() as p:
		t0 = time.time()
		results = p.starmap(run_iteration,args)
		import json

		with open("saved_states.txt","w") as file:
			file.write(json.dumps(results))
		print(f"ran in {(time.time()-t0):.2f}s")
