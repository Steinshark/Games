
# All pytorch modules 
import torch
import torch.nn as nn
from torch.nn import Conv2d,Linear,Flatten,ReLU

#All supporting modules 
import random
import numpy 
import time 
from networks import ConvolutionalNetwork
import copy
#This class interfaces only with NP 
class Snake:


	#	CONSTRUCTOR 
	#	This method initializes the snake games to be played until each are over 
	#	i.e. it allows for all 16, 32, etc... games of a batch to be played at once.
	def __init__(self,w,h,learning_model:nn.Module,simul_games=32,memory_size=4,device=torch.device('cuda'),rewards={"die":-1,'eat':1,"step":-.01},max_steps=200):


		#Set global Vars
		#	grid_w maintains the width of the game field 
		#	grid_h tracks the height of the game field 
		#	simul_games is how many games will be played at once. 
		# 	- essential this is the batch size  
		self.grid_w 			= w
		self.grid_h 			= h
		self.simul_games 		= simul_games
		self.cur_step			= 0

		#	GPU or CPU computation are both possible 
		#	Requires pytorch for NVIDIA GPU Computing Toolkit 11.7
		self.device 			= device


		#	Hopefully is a CNN 
		# 	must be a torch.nn Module
		self.learning_model 	= learning_model
		self.learning_model.to(device)


		#	This list holds information on each game, as well as their experience sets.
		#	Since all games are played simultaneously, we must keep track of which ones
		#	should still be played out and which ones are over.
		#
		#	MAINTAINS:
		#		status: 		if the game is over or not 
		#		experiences: 	the set of (s,a,s`,r,done) tuples 
		#		highscore:		how many foods this snake collected
		#		lived_for:		how many steps this snake survived
		self.game_collection 	= [{"status":"active","experiences":[],"highscore":0,"lived_for":0,"eaten_since":0,"step_reward":None} for _ in range(simul_games)]
		self.active_games 		= list(range(simul_games))

		#	Games are held in a 4D numpy array 
		#   (GAME#,CHANNEL#,HEIGHT,WIDTH)
		#	
		#	CHANNELS:
		#	0: one-hot encodes the snake's head location 
		#	1: one-hot encodes the snake's body's locations 
		#	2: one-hot encodes the food's location 
		#	REPEAT memory_size# times. i.e. memory_size = 4 means 12 channels
		self.game_vectors 		= torch.zeros(size=(simul_games,memory_size*3,h,w),dtype=torch.float,device=device,requires_grad=False)

		#	The game directions are tracked in this list. 
		# 	Each tuple encodes the step taken in (x,y).
		#
		# 	Tuples should only ever have one '1' in it. 
		# 	i.e. (1,1) & (0,0) are NOT legal direction tuples
		self.direction_vectors 	= [ random.randint(0,3) for _ in range(simul_games) ]
		self.prev_dir_vectors 	= self.direction_vectors
		#	The food directions are tracked in this list. 
		# 	Each tuple encodes the food at pos (x,y).
		self.food_vectors 		= [ [random.randint(0,w-1),random.randint(0,w-1)] for _ in range(simul_games) ]

		#	The snake is tracked in this list 
		#	Used to update Numpy Arrays in a more efficient manner.
		self.snake_tracker		= [ [[0,0]] for _ in range(simul_games) ]
		self.full_game_tracker 	= [[] for _ in range(simul_games)]

		#	Store all experiences in a list of 
		#	dictionaries that will be returned to the training class
		self.experiences 		= list()
		
		#	A very important hyper-parameter: the reward made for each action
		self.reward 			= rewards 
		self.move_threshold  	= max_steps
		self.movements 			= [(0,-1),(0,1),(-1,0),(1,0)]



	#	GAME PLAYER 
	#	Calling this method will play out all games until completion
	def play_out_games(self,epsilon=.2,debugging=False):
		
		#	Maintain some global parameters 
		self.cur_step = 0

		for snake_i,pos in enumerate(self.food_vectors):
			x,y = pos[0],pos[1]
			self.game_vectors[snake_i,2,y,x] = 1
		#	Spawn the snake in a random location each time
		for snake_i in range(self.simul_games):
			game_start_x,game_start_y = random.randint(0,self.grid_w-1),random.randint(0,self.grid_h-1)
			self.game_vectors[snake_i,0,game_start_y,game_start_x] = 1
			self.snake_tracker[snake_i] = [[game_start_x,game_start_y]]


		#	Game Loop executes while at least one game is still running 
		#	Executes one step of the game and does all work accordingly
		while True:			
			#	GET NEXT DIR  
			#	- an epsilon-greedy implementation 
			#	- choose either to exploit or explore based off of 
			#	  some threshold. At first, P(explore) >> P(exploit).
			#	- decides for ALL GAMES simultaneously
			
			# 	The model for this dropoff will probably change and is 
			#	open to exploration
			for snake_i in self.active_games:
				self.full_game_tracker[snake_i].append({"snake":self.snake_tracker[snake_i],'food':self.food_vectors[snake_i]})

			if random.random() < epsilon:
				self.explore()
			else:
				self.exploit()
			
			#	MAKE NEXT MOVES 
			#	Involves querying head of each game, finding where it will end next,
			#	and applying game logic to kill/reward it 
			
			#	Step
			self.step_snake()
			 
			# 	Check if we are done 
			if len(self.active_games) == 0:
				return self.cleanup()
			else:
				self.cur_step+=1
			
		return
	





	#############################################################
	#															#
	#	HELPER FUNCTIONS TO MAKE TRAINING FUNCTION LOOK NICER   #
	#															#
	 
	#	EXPLORE 
	# 	Update all directions to be random.
	#	This includes illegal directions i.e. direction reversal
	def explore(self):
		self.prev_dir_vectors = copy.copy(self.direction_vectors)

		for snake_i in self.active_games:
			cur_dir = self.direction_vectors[snake_i]

			#Give it only legal moves
			if cur_dir in [0,1]:
				self.direction_vectors[snake_i] = random.randint(2,3)
			elif cur_dir == [2,3]:
				self.direction_vectors[snake_i] = random.randint(0,1)
			
			#self.direction_vectors[snake_i] = random.randint(0,3) 
	


	#	EXPLOIT 
	# 	Sends ALL games into model to be predicted (probably faster than sifting (???))
	def exploit(self,mode='Alive'):

		self.prev_dir_vectors = copy.copy(self.direction_vectors)
		#	Inputs are of shape (#Games,#Channels,Height,Width) 
		#	Model output should be of shape (#Games,4)
		#	model_out corresponds to the EV of taking direction i for each game
		
		if mode == 'All':
			with torch.no_grad():
				model_out = self.learning_model.forward(self.game_vectors)
				#Find the direction for each game with highest EV 
				next_dirs = torch.argmax(model_out,dim=1)

				#Update direction vectors accordingly 
				self.direction_vectors = next_dirs
		
		elif mode == 'Alive':
			with torch.no_grad():
				for snake_i in self.active_games:
					model_out 	= self.learning_model.forward(self.game_vectors.narrow(0,snake_i,1))
					next_dir 	= torch.argmax(model_out)
					self.direction_vectors[snake_i] = next_dir

	#	STEP SNAKE 
	#	Move each snake in the direction that dir points 
	#	Ensure we only touch active games
	def step_snake(self):

		mark_del = []
		i = self.active_games[0]
		for snake_i in self.active_games:



			# DEBUG 
			#if snake_i == i and  print(f"snake {i} - {self.snake_tracker[i]}\ninit dir {self.movements[self.direction_vectors[i]]}\ninit food {self.food_vectors[i]}\ninit state:\n{self.game_vectors[snake_i]}"): pass
			

			#	Find next location of snake 
			chosen_action = self.direction_vectors[snake_i]
			dx,dy = self.movements[chosen_action]
			next_x = self.snake_tracker[snake_i][0][0]+dx
			next_y = self.snake_tracker[snake_i][0][1]+dy
			next_head = [next_x,next_y]
			
			#Check if this snake lost 
			if next_x < 0 or next_y < 0 or next_x == self.grid_w or next_y == self.grid_h or next_head in self.snake_tracker[snake_i] or self.game_collection[snake_i]['eaten_since'] > self.move_threshold or self.check_opposite(snake_i):
				
				#Mark for delete and cleanup
				mark_del.append(snake_i)
				self.game_collection[snake_i]['status'] = "done"
				self.game_collection[snake_i]['highscore'] = len(self.snake_tracker[snake_i])-1
				self.game_collection[snake_i]["lived_for"] = self.cur_step

				#Add final experience
				experience = {"s":self.game_vectors.narrow(0,snake_i,1).clone(),"a":chosen_action,"r":self.reward['die'],'s`':self.game_vectors.narrow(0,snake_i,1).clone(),'done':0}
				
				#Dont penalize fully for threshold
				if self.game_collection[snake_i]['eaten_since'] > self.move_threshold:
					experience['r'] = self.reward['step']

				self.experiences.append(experience)
				continue
			
			#	START EXP CREATION 	
			experience = {"s":self.game_vectors.narrow(0,snake_i,1).clone(),"a":chosen_action,"r":None,'s`':self.game_vectors.narrow(0,snake_i,1).clone(),'done':1}
			
			#	ROLL VECTORS 
			#	Since the snake has survived, we can roll 3 channels down to be written with 
			#	the new snake state .
			self.game_vectors[snake_i] = torch.roll(self.game_vectors[snake_i],3,dims=0)
			self.game_vectors[snake_i][0:3] = torch.zeros(size=(3,self.grid_h,self.grid_w),dtype=torch.float,device=self.device,requires_grad=False)

			#Check if snake ate food
			if next_head == self.food_vectors[snake_i]:
				
				#Change location of the food
				self.spawn_new_food(snake_i)
				
				#	Mark snake to grow by 1 (keep the last snake segment)
				#	in both the snake tracker and game vector 
				snake_tail_x,snake_tail_y = self.snake_tracker[snake_i][-1][0],self.snake_tracker[snake_i][-1][1]
				self.game_vectors[snake_i][1][snake_tail_y][snake_tail_x] = 1
				self.snake_tracker[snake_i].append(self.snake_tracker[snake_i][-1])

				#Set snake reward to be food 
				experience['r'] = self.reward['eat']
				self.game_collection[snake_i]["eaten_since"] = 0
			
			
			else:
				experience['r'] = self.reward["step"]
			
			#Grow the snake by 1 step
			self.game_vectors[snake_i][0][next_y][next_x] = 1

			#	Append the rest of the body up till the last segment
			#	if the snake was meant to grow, then the final segment 
			#	will have been added in the above conditional block
			for snake_body_pos in self.snake_tracker[snake_i][:-1]:
				x,y = snake_body_pos[0],snake_body_pos[1] 
				self.game_vectors[snake_i][1][y][x] = 1
			
			#Counts the head in the body channel 
			self.game_vectors[snake_i][1][next_head[1]][next_head[0]] = 1

			#Update the food location vector 
			food_x,food_y = self.food_vectors[snake_i]
			self.game_vectors[snake_i][2][food_y][food_x] = 1
			self.game_collection[snake_i]["eaten_since"] += 1

			#	Update snake tracker with some finicky magic 
			#	If the snake grew, then snake tracker will have been made artificially longer 
			# 	to account for growth 
			self.snake_tracker[snake_i] = [next_head] + self.snake_tracker[snake_i][:-1]


			#	Add s` to the experience 
			experience['s`'] = self.game_vectors.narrow(0,snake_i,1).clone()
			self.experiences.append(experience)
			

		#Delete all items from mark_del  
		for del_snake_i in mark_del:
			self.active_games.remove(del_snake_i)
	
	
		#DEBUG 
		# s_i = self.active_games[0]
		# print(f"Game #{s_i}")
		# print(f"moved:{self.movements[self.direction_vectors[s_i]]}")
		# print(f"Snake is: {self.snake_tracker[s_i]}")
		# input(self.game_vectors[s_i])



		return 



	#	SPAWN NEW FOOD 
	#	Place a random food on map.
	#	Check that its not in the snake
	#	Repeat until above is True
	def spawn_new_food(self,snake_i):
		next_x = random.randint(0,self.grid_w-1)
		next_y = random.randint(0,self.grid_h-1)
		food_loc = [next_x,next_y]

		while food_loc in self.snake_tracker[snake_i]:
			next_x = random.randint(0,self.grid_w-1)
			next_y = random.randint(0,self.grid_h-1)
			food_loc = [next_x,next_y] 

		self.food_vectors[snake_i] = food_loc 
		self.game_vectors[snake_i][2][next_y][next_x] = 1
		return next_x,next_y



	#	SAVE ALL GAME EXPS
	#	Apply game logic to see which snakes died,
	# 	which eat, and which survive 
	def cache_round(self):
		return


	#	RETURN TO TRAINER
	def cleanup(self):
		return self.game_collection,self.experiences,self.full_game_tracker

	def check_opposite(self,snake_i):
		if self.cur_step == 0 :
			return False
		dir_1 = self.direction_vectors[snake_i]
		dir_2 = self.prev_dir_vectors[snake_i]

		return abs(dir_1-dir_2) == 1 and not dir_1+dir_2 == 3
		
if __name__ == "__main__":
	w = 3
	h = 3
	model 	= ConvolutionalNetwork(loss_fn=torch.nn.HuberLoss,optimizer_fn=torch.optim.Adam,lr=.0001,wd=0,architecture=[Conv2d(6,16,3,1,1) ,ReLU(),Conv2d(16,32,3,1,0),ReLU(),Flatten(),Linear(128,128),ReLU(),Linear(128,4)],input_shape=(1,6,w,h))
	s 		= Snake(w,h,model,device=torch.device("cpu"),memory_size=2)

	model.forward(torch.ones(size=(1,6,w,h)))
	#t0 = time.time()
	#s.play_out_games()


	for i in range(6):
		s.exploit()
		s.step_snake()
		if len(s.active_games) == 0:
			print("all snakes died")
			print(s.direction_vectors)
			break
		print(f"GAME REPR:")
		print(s.game_vectors[s.active_games[0]])
		print(f"game:{s.active_games[0]}")
		print(f"dir:{s.movements[s.direction_vectors[s.active_games[0]]]}")
		print(f"snake:{s.snake_tracker[s.active_games[0]]}")
		input(f"status:{s.game_collection[s.active_games[0]]['status']}")
	#print(f"list took {(time.time()-t0)}")
	
	# s = Snake(13,13,None)
	# s.active_games = {i:True for i in range(32)}
	# t0 = time.time()
	# for i in range(100000):
	# 	s.explore()
	# 	s.step_snake()
	# print(f"dict took {(time.time()-t0)}")