'''
AUTHOR:     Everett Stenberg 
GITHUB:     Steinshark
UPDATED:    12NOV22

Format for comments is:
CLASS SEPARATOR:        3 lines 
METHOD SEPARATOR:       2 lines
OBV BLOCK SEPARATOR:    1 line 
'''



import pygame 
import time 
import torch 
from torch import nn

#Proprietary code 
import networks



#A trainer is used in conjunction with a program to optimize (the game) to run 
#the entire operation from start to end. Responsible for instantiating games,
#building the exp set, and training a model to optimally play that game
class Trainer:


	#Trainer creates an environment to run n iterations of snake game in
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


	#"Train" is where the work is done. Methods "train_on_experiences" and "transer_models" are 
	#called from here. "train" plays "episodes" games and obeys the hyperparameters specivied in kwargs
	def train(self,episodes=1000,train_every=1000,replay_buffer=32768,sample_size=128,batch_size=32,epochs=10,early_stopping=True,transfer_models_every=2000,verbose=True):

		self.high_score = 0
		self.best = 0
		clear_every = 2
		experiences = []
		replay_buffer_size = replay_buffer
		t0 = time.time()
		high_scores = []
		trained = False

		scores = []
		lived = []

		picking = False
		for e_i in range(int(episodes)):

			#Play a game and collect the experiences
			game = SnakeGame(self.w,self.h,fps=100000,encoding_type=self.encoding_type,device=self.device)
			exp, score,lived_for = game.train_on_game(self.learning_model,visible=False,epsilon=self.epsilon)

			scores.append(score+1)
			lived.append(lived_for)

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
					print(f"[Episode {str(e_i).rjust(len(str(episodes)))}/{int(episodes)}  -  {(100*e_i/episodes):.2f}% complete\t{(time.time()-t0):.2f}s\te: {self.epsilon:.2f}\thigh_score: {self.high_score}] lived_avg: {sum(lived[-1000:])/len(lived[-1000:]):.2f} score_avg: {sum(scores[-1000:])/len(scores[-1000:]):.2f}")
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
		lived = [sum(lived[i:i+smooth])/smooth for i in range(0,int(len(lived)),smooth)]


		#Save a fig for results

		#Top plot for avg. score, bottom plot for avg. time lived
		fig, axs = plt.subplots(2,1)
		fig.set_size_inches(19.2,10.8)

		#Plot data
		axs[0].plot([i*smooth for i in range(len(scores))],scores,label="scores",color='green')
		axs[1].plot([i*smooth for i in range(len(lived))],lived,label="lived for",color='cyan')
		axs[0].legend()
		axs[1].legend()
		axs[0].set_title(f"{self.architecture}-{str(self.loss_fn).split('.')[-1][:-2]}-{str(self.optimizer_fn).split('.')[-1][:-2]}-ep{epochs}-lr{self.lr}-bs{batch_size}-te{train_every}-rb{replay_buffer}-ss{sample_size}")

		#Save fig to figs directory
		if not os.path.isdir("figs"):
			os.mkdir("figs")
		fig.savefig(os.path.join("figs",f"{self.architecture}-{str(self.loss_fn).split('.')[-1][:-2]}-{str(self.optimizer_fn).split('.')[-1][:-2]}-ep{epochs}-lr{self.lr}-bs{batch_size}-te{train_every}-rb{replay_buffer}-ss{sample_size}.png"),dpi=100)


		#Return the best score, high scores of all episode blocks, scores, and steps lived
		return self.best,high_scores,scores,lived


	#Used tp train a batch of samples built in the "train" method
	def train_on_experiences(self,big_set,epochs=100,batch_size=8,early_stopping=True,verbose=False):
		
		for epoch_i in range(epochs):
			t0 = time.time()
			#Printing things
			if verbose and print(f"EPOCH {epoch_i}:\n\t",end='training['): pass
			next_percent = .02

			#Batch the sample set
			batches = [[big_set[i * n] for n in range(batch_size)] for i in range(int(len(big_set)/batch_size))]

			#Measure losses and prepare for early stopping
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

				loss = self.learning_model.loss_fn(vals_target_adjusted,predictions)
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

	#Used in 2 network model of RL (DDQN)
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


#An AcceleratedTrainer allows the use of a GPU to accelerate computation, though 
#perhaps minimally in some scenarios. In cases involving training on large batch sizes,
#an AcceleratedTrainer helps more.
class AcceleratedTrainer(Trainer):

	def __init__(self):
		super().__init__()
