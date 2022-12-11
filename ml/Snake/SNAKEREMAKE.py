
# All pytorch modules 
import torch
import torch.nn as nn

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
	def __init__(self,w,h,learning_model:nn.Module,simul_games=32,memory_size=4,device=torch.device('cuda'),rewards={"die":-5,"food":5,"step":0}):


		#Set global Vars
		#	grid_w, grid_ maintains the height width of the game field 
		#	img_w, img_h maintain the size of the imgs that are used as state representations 
		#	
		#	simul_games is how many games will be played at once. 
		# 	- essential this is the batch size  
		self.grid_w 				= w
		self.grid_h 				= h

		#self.img_w				= 48		#(allows for grid sizes {1,2,3,4,6,8,12,16,24})
		#self.img_h				= 48		#(allows for grid sizes {1,2,3,4,6,8,12,16,24})
		self.HEAD_CH				= 0
		self.BODY_CH				= 1
		self.FOOD_CH				= 2
		#BLOCK_FACT_Y			= int(self.img_w / self.grid_w)
		#BLOCK_FACT_X			= int(self.img_w / self.grid_w)

		self.simul_games 			= simul_games
		self.cur_step				= 0

		self.channels = memory_size*3
		#	GPU or CPU computation are both possible 
		#	Requires pytorch for NVIDIA GPU Computing Toolkit 11.7
		self.device 				= device


		#	Hopefully is a CNN 
		# 	must be a torch.nn Module
		self.learning_model 		= learning_model
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
		self.games 					= [{"eaten_since":0,'snake':[(0,0)],"dir":0,"prev_dir":0,"food":(0,0),"exp":{}} for _ in range(simul_games)]
		self.cur_states				= torch.zeros(size=(simul_games,self.channels,h,w),requires_grad=False,device=device)
		self.active_games 			= list(range(simul_games))
		self.live_game_model_inputs = torch.zeros(size=(simul_games,self.channels,h,w),requires_grad=False,device=device)

		#	Store all experiences in a list of 
		#	dictionaries that will be returned to the training class
		self.experiences 			= list()
		self.stats					= {"scores":[],"lives":[]}
		#	A very important hyper-parameter: the reward made for each action
		self.reward 				= rewards 
		self.move_threshold  		= self.grid_w * self.grid_h * 2
		self.movements 				= [(0,-1),(0,1),(-1,0),(1,0)]



	#	GAME PLAYER 
	#	Calling this method will play out all games until completion
	def play_out_games(self,epsilon:float=.2,debugging:bool=False):
		
		#	Maintain some global parameters 
		self.cur_step = 0

		#	Spawn the food and snake in a random location each time
		for snake_i in self.active_games:

			#Get random snake location 
			start_x,start_y = random.randint(0,self.grid_w-1),random.randint(0,self.grid_h-1)
			self.games[snake_i]['snake'] = [(start_x,start_y)]
			
			#Get random food loc 
			self.spawn_new_food(snake_i)
			

		#	Game Loop executes while at least one game is still running 
		#	Executes one step of the game and does all work accordingly
		while True:		
			
			for snake_i in self.active_games:
				self.games[snake_i]['exp'] = {}
			#	Get repr of all live games 
			#	This will be fed into the model 
			#	as well as be used for the exp 
			self.get_state_reprs('s',roll=self.cur_step==0)										#[S: [X]   A: [ ]	S`: [ ]	R:[ ] D: [ ]]
			#	GET NEXT DIR  
			#	- an epsilon-greedy implementation 
			#	- choose either to exploit or explore
			if random.random() < epsilon:
				self.explore()
			else:
				self.exploit()																		#[S: [X]   A: [X]	S`: [ ]	R:[ ] D: [ ]]
			
			#	MAKE NEXT MOVES 
			#	Involves querying head of each game, finding where it will end next,
			#	and applying game logic to kill/reward it 	
			self.step_snake()																		#[S: [X]   A: [X]	S`: [X]	R:[X] D: [X]]

			# 	Check if we are done 
			if len(self.active_games) == 0:
				return self.cleanup()
			else:
				self.cur_step+=1
			
		
	





	#############################################################
	#															#
	#	HELPER FUNCTIONS TO MAKE TRAINING FUNCTION LOOK NICER   #
	#															#
	 
	#	EXPLORE 
	# 	Update all directions to be random.
	#	This includes illegal directions i.e. direction reversal
	def explore(self):


		for snake_i in self.active_games:
			self.games[snake_i]['prev_dir'] 	= self.games[snake_i]['dir']
			self.games[snake_i]['dir'] 			= random.randint(0,3) 
			self.games[snake_i]['exp']['a']		= self.games[snake_i]['dir']
	


	#	EXPLOIT 
	# 	Asks model for "the best" move
	def exploit(self):
		
		#Get best from model
		model_outputs = self.learning_model.model(self.live_game_model_inputs)
		chosen_dirs   =	torch.argmax(model_outputs,dim=1)

		#Save old dir 
		for i_eq,snake_i in enumerate(self.active_games):
				self.games[snake_i]['prev_dir'] = self.games[snake_i]['dir']	
				self.games[snake_i]['dir'] 		= chosen_dirs[i_eq]
				self.games[snake_i]['exp']['a'] = chosen_dirs[i_eq]



	#	STEP SNAKE 
	#	Move each snake in the direction that dir points 
	#	Ensure we only touch active games
	def step_snake(self):

		mark_del = []
		
		#DEBUG
		# i = self.active_games[0]
		# #/DEBUG
		# print(f"active games{self.active_games}")
		for snake_i in self.active_games:



			#DEBUG 
			#if snake_i == i and  print(f"snake {i} - {self.snake_tracker[i]}\ninit dir {self.movements[self.direction_vectors[i]]}\ninit food {self.food_vectors[i]}\ninit state:\n{self.game_vectors[snake_i]}"): pass
			#/DEBUG

			#	Find next location of snake 
			dx,dy = self.movements[self.games[snake_i]['dir']]
			next_x = self.games[snake_i]['snake'][0][0]+dx
			next_y = self.games[snake_i]['snake'][0][1]+dy
			next_head = (next_x,next_y)
			

			#	LOSE CASE
			if next_x < 0 or next_y < 0 or next_x == self.grid_w or next_y == self.grid_h or next_head in self.games[snake_i]['snake'] or self.games[snake_i]['eaten_since'] > self.move_threshold or self.check_opposite(snake_i):
				
				#Mark for delete and record stats
				mark_del.append(snake_i)

				self.stats['scores'].append(len(self.games[snake_i])-1)
				self.stats['lives'].append(self.cur_step)

				#Add final experience
				exp = self.games[snake_i]['exp']

				exp['r'] 	= self.reward['die']
				exp['s`'] 	= None
				exp['done']	= True

				self.experiences.append(exp)
				continue
			
			
			#	EAT CASE
			if next_head == self.games[snake_i]['food']:
				
				#Change location of the food
				self.spawn_new_food(snake_i)
				
				#	Mark snake to grow by 1 (keep the last snake segment)
				self.games[snake_i]['snake'].insert(1,next_head)	 
				
				#Set snake reward to be food 
				self.games[snake_i]['exp']['r'] = self.reward['food']
				self.games[snake_i]['exp']['done'] = False
				self.games[snake_i]["eaten_since"] = 0
			

			#	SURVIVE CASE
			else:
				self.games[snake_i]['snake'] = [next_head] + self.games[snake_i]['snake'][:-1]	
				self.games[snake_i]['exp']['r'] = self.reward["step"]
				self.games[snake_i]['exp']['done'] = False

				self.games[snake_i]["eaten_since"] += 1

		#
		# print(f"removing games{mark_del}")
		#	Delete all dead snakes  
		for del_snake_i in mark_del:
			self.active_games.remove(del_snake_i)
		
		#	Get all reprs of remaining snakes 
		# 	Add s` to their experience set  
		self.get_state_reprs('s`')
		
		#	Add all experiences
		self.experiences.extend([item['exp'] for item in self.games])
		
		#DEBUG 
		# s_i = self.active_games[0]
		# print(f"snake: {s_i}")
		# print(f"moved:{self.games[i]['dir']} -> {self.movements[self.games[i]['dir']]}")
		# print(self.games[i]['snake'])
		# print(self.games[i]['exp']['s'])
		# print(self.games[i]['exp']['r'])
		# print(self.games[i]['exp']['done'])
		# input(self.games[i]['exp']['s`'])
		#/DEBUG



		return 



	#	SPAWN NEW FOOD 
	#	Place a random food on map.
	#	Check that its not in the snake
	#	Repeat until above is True
	def spawn_new_food(self,snake_i:int):
		next_x = random.randint(0,self.grid_w-1)
		next_y = random.randint(0,self.grid_h-1)
		food_loc = (next_x,next_y)

		while food_loc == self.games[snake_i]['food']:
			next_x = random.randint(0,self.grid_w-1)
			next_y = random.randint(0,self.grid_h-1)
			food_loc = (next_x,next_y) 

		self.games[snake_i]['food'] = food_loc 

 
	#	RETURN TO TRAINER
	#	Sends the final stats 
	#	and experiences to the trainer
	def cleanup(self):
		return self.game_collection,self.experiences



	#	Check if the snake turned 180
	#	Keeps code slightly cleaner
	def check_opposite(self,snake_i:int):
		dir_1 = self.games[snake_i]['prev_dir']
		dir_2 = self.games[snake_i]['dir']

		return dir_1-dir_2 == 1 or dir_2-dir_1 == 1 and not dir_1+dir_2 == 3
	
	

	#	Vectorizes the snake 
	#	Goes through all live games 
	#	And turns the snakes into a
	#	4D numpy array
	def	get_state_reprs(self,state,roll=True):
		
		# if rolling_states:
		# 	torch.roll(self.live_game_model_inputs,3,dims=1)
		# 	self.live_game_model_inputs[:,0:3] = self.cur_states

		#Get all still active games
		if not roll:
			for snake_i in self.active_games:
				self.games[snake_i]['exp'][state] = self.cur_states[snake_i].clone()
			return 
		for i_eq,snake_i in enumerate(self.active_games):
			self.cur_states[snake_i] = torch.roll(self.cur_states[snake_i],3,dims=0)
			self.cur_states[snake_i][0:3] = torch.zeros(size=(3,self.grid_h,self.grid_w))
			#HEAD 
			head_x,head_y = self.games[snake_i]['snake'][0] 
			self.cur_states[snake_i,self.HEAD_CH,head_y,head_x] = 1

			#BODY 
			for segment in self.games[snake_i]['snake'][1:]:
				x,y = segment
				self.cur_states[snake_i,self.BODY_CH,y,x] = 1
			
			#Food 
			food_x,food_y = self.games[snake_i]['food']
			self.cur_states[snake_i,self.FOOD_CH,food_y,food_x] = 1

			#Add to the snake's cur exp
			self.games[snake_i]['exp'][state] = self.cur_states[snake_i].clone()

		if self.active_games:
			self.live_game_model_inputs = torch.stack([self.cur_states[snake_i] for snake_i in self.active_games])













if __name__ == "__main__":
	w = 4 
	h = 4
	mem=2
	model 	= ConvolutionalNetwork(loss_fn=torch.nn.HuberLoss,optimizer_fn=torch.optim.Adam,lr=.0001,wd=0,architecture=[[3*mem,4,3],[144,4]],input_shape=(1,12,4,4))
	s = Snake(w,h,model,memory_size=mem)
	#t0 = time.time()
	s.play_out_games()
	#print(f"list took {(time.time()-t0)}")
	
	# s = Snake(13,13,None)
	# s.active_games = {i:True for i in range(32)}
	# t0 = time.time()
	# for i in range(100000):
	# 	s.explore()
	# 	s.step_snake()
	# print(f"dict took {(time.time()-t0)}")