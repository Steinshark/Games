
# All pytorch modules 
import torch
import torch.nn as nn

#All supporting modules 
import random
import numpy 
import time 


#This class interfaces only with NP 
class Snake:


	#	CONSTRUCTOR 
	#	This method initializes the snake games to be played until each are over 
	#	i.e. it allows for all 16, 32, etc... games of a batch to be played at once.
	def __init__(self,w,h,learning_model:nn.Module,simul_games=32,memory_size=4,device=torch.device('cpu'),rewards={"die":-1,"food":1,"step":-.01}):


		#Set global Vars
		#	grid_w maintains the width of the game field 
		#	grid_h tracks the height of the game field 
		#	simul_games is how many games will be played at once. 
		# 	- essential this is the batch size  
		self.grid_w 			= w
		self.grid_h 			= h
		self.simul_games 		= simul_games


		#	GPU or CPU computation are both possible 
		#	Requires pytorch for NVIDIA GPU Computing Toolkit 11.7
		self.device 			= device


		#	Hopefully is a CNN 
		# 	must be a torch.nn Module
		self.learning_model 	= learning_model

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
		self.game_vectors 		= numpy.zeros(shape=(simul_games,memory_size*3,h,w))

		#	The game directions are tracked in this list. 
		# 	Each tuple encodes the step taken in (x,y).
		#
		# 	Tuples should only ever have one '1' in it. 
		# 	i.e. (1,1) & (0,0) are NOT legal direction tuples
		self.direction_vectors 	= [ [1,0] for _ in range(simul_games) ]

		#	The food directions are tracked in this list. 
		# 	Each tuple encodes the food at pos (x,y).
		self.food_vectors 		= [ [random.randint(0,w-1),random.randint(0,w-1)] for _ in range(simul_games) ]

		#	The head and tail of each snake is tracked in this list 
		#	Used to update Numpy Arrays in a more efficient manner.
		#	I.e. only head and tail must be moved each game step
		self.snake_tracker		= [ [[0,0]] for _ in range(simul_games) ]


		#	A very important hyper-parameter: the reward made for each action
		self.reward = rewards 
		self.movements = [[0,-1],[0,1],[-1,0],[1,0]]

	#	GAME PLAYER 
	#	Calling this method will play out all games until completion
	def play_out_games(self,epsilon=.2):
		
		#	Maintain some global parameters 
		
		cur_step = 0

		#	Spawn the snake in a random location each time
		game_start_x,game_start_y = random.randint(0,self.grid_h-1),random.randint(0,self.grid_w-1)
		self.game_vectors[:,1,game_start_y,game_start_x] = 1
		self.snake_tracker = [[game_start_x,game_start_y] for _ in range(self.simul_games)]

		#	Game Loop executes while at least one game is still running 
		#	Executes one step of the game and does all work accordingly
		while len(self.active_games) > 0:
			pass
			

			#	GET NEXT DIR  
			#	- an epsilon-greedy implementation 
			#	- choose either to exploit or explore based off of 
			#	  some threshold. At first, P(explore) >> P(exploit).
			#	- decides for ALL GAMES simultaneously
			
			# 	The model for this dropoff will probably change and is 
			#	open to exploration
			if random.random() < epsilon:
				self.explore()
			else:
				self.exploit()
			
			#	MAKE NEXT MOVES 
			#	Involves querying head of each game, finding where it will end next,
			#	and applying game logic to kill/reward it 
			
			#	Step
			self.step_snake()

		# 	#Add to experiences

		# 	experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`': self.get_state_vector()})

		# 	eaten_since += 1

		# 	#Check if lived too long
		# 	if eaten_since > self.grid_w*self.grid_h*2:
		# 		reward = rewards['idle']
		# 		experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`':self.get_state_vector()})
		# 		return experiences, score, lived
		# return experiences, score, lived
		return
	


	#############################################################
	#															#
	#	HELPER FUNCTIONS TO MAKE TRAINING FUNCTION LOOK NICER   #
	#															#
	 
	#	EXPLORE 
	# 	Update all directions to be random.
	#	This includes illegal directions i.e. direction reversal
	def explore(self):
		self.direction_vectors = [self.movements[random.randint(0,3)] for _ in range(self.simul_games)]
	
	#	EXPLOIT 
	# 	Sends ALL games into model to be predicted (probably faster than sifting (???))
	def exploit(self):


		#	Inputs are of shape (#Games,#Channels,Height,Width) 
		#	Model output should be of shape (#Games,4)
		#	model_out corresponds to the EV of taking direction i for each game
		model_out = self.learning_model(self.game_vectors)

		#Find the direction for each game with highest EV 
		next_dirs = torch.argmax(model_out,dim=1)

		#Update direction vectors accordingly 
		self.direction_vectors = [self.movements[dir_i] for dir_i in next_dirs]


	#	STEP SNAKE 
	#	Move each snake in the direction that dir points 
	#	Ensure we only touch active games
	def step_snake(self):

		mark_del = []
		
		for snake_i in self.active_games:
			next_x = self.snake_tracker[snake_i][0][0]+self.direction_vectors[snake_i][0]
			next_y = self.snake_tracker[snake_i][0][1]+self.direction_vectors[snake_i][1]
			next_head = [next_x,next_y]

			#Check if this snake lost 
			if next_x < 0 or next_y < 0 or next_x == self.grid_w or next_y == self.grid_h or next_head in self.snake_tracker[snake_i]:
				
				#Mark for delete and cleanup
				mark_del.append(snake_i)
				self.game_collection[snake_i]['status'] = "done"

				#Add final experience
				self.game_collection[snake_i]["step_reward"] = self.reward['die']
				continue

			#	ROLL VECTORS 
			#	Since the snake has survived, we can roll 3 channels down to be written with 
			#	the new snake state .
			self.game_vectors[snake_i] = numpy.roll(self.game_vectors[snake_i],3,axis=0)
			self.game_vectors[snake_i][0:3] = numpy.zeros(shape=(3,self.grid_h,self.grid_w))

			#Check if snake ate food
			if next_head == self.food_vectors[snake_i]:

				#Change location of the food
				self.spawn_new_food(snake_i)
				
				#Mark snake to grow by 1 (keep the last snake segment)
				snake_tail_x,snake_tail_y = self.snake_tracker[snake_i][-1][0],self.snake_tracker[snake_i][-1][1]
				self.game_vectors[snake_i][snake_tail_y][snake_tail_x] = 1
				self.snake_tracker[snake_i].append(self.snake_tracker[snake_i][-1])

				#Set snake reward to be food 
				self.game_collection[snake_i]['step_reward'] = self.reward['food']

			#Grow the snake by 1 step
			self.game_vectors[snake_i][0][next_y][next_x] = 1

			#	Append the rest of the body up till the last segment
			#	if the snake was meant to grow, then the final segment 
			#	will have been added in the above conditional block
			for snake_body_pos in self.snake_tracker[snake_i][:-1]:
				x,y = snake_body_pos[0],snake_body_pos[1] 
				self.game_vectors[snake_i][1][y][x] = 1
			
			#Update the food location vector 
			food_x,food_y = self.food_vectors[snake_i]
			self.game_vectors[snake_i][2][food_y][food_x] = 1

			#	Update snake tracker with some finicky magic 
			#	If the snake grew, then snake tracker will have been made artificially longer 
			# 	to account for growth 
			self.snake_tracker[snake_i] = [next_head] + self.snake_tracker[snake_i][:-1]

		#Delete all items from mark_del  
		for del_snake_i in mark_del:
			self.active_games.remove(del_snake_i)
		
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
		return next_x,next_y

	#	CHECK SNAKE
	#	Apply game logic to see which snakes died,
	# 	which eat, and which survive 
	def check(self):
		next_neads = 0

if __name__ == "__main__":
	s = Snake(4,4,None)
	t0 = time.time()
	for i in range(100000):
		s.explore()
		s.step_snake()
		print(s.game_vectors[0][:3])
		print(f"dir:{s.direction_vectors[0]}")
		print(f"snake:{s.snake_tracker[0]}")
		input(f"status:{s.game_collection[0]['status']}")
	print(f"list took {(time.time()-t0)}")
	
	# s = Snake(13,13,None)
	# s.active_games = {i:True for i in range(32)}
	# t0 = time.time()
	# for i in range(100000):
	# 	s.explore()
	# 	s.step_snake()
	# print(f"dict took {(time.time()-t0)}")