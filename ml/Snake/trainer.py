import networks 
import torch 
import os 
import time 
import SnakeConcurrent
import random 
from matplotlib import pyplot as plt 
import numpy 
import sys 
import tkinter as tk
from snakeAI import SnakeGame
from telemetry import plot_game
class Trainer:

	def __init__(	self,game_w,game_h,
					visible=True,
					loading=True,
					PATH="models",
					memory_size=4,
					loss_fn=torch.nn.MSELoss,
					optimizer_fn=torch.optim.Adam,
					lr=5e-4,
					wd=0,
					fname="experiences",
					name="generic",
					gamma=.96,
					architecture=[256,32],
					gpu_acceleration=False,
					epsilon=.2,
					m_type="CNN",
					save_fig_now=False,
					progress_var=None,
					output=sys.stdout,
					steps=None,
					scored=None,
					score_tracker=[],
					step_tracker=[],
					parent_instance=None,
					game_tracker=[],
					gui=False):

		self.PATH = PATH
		self.fname = fname
		self.name = name 
		self.m_type = m_type
		self.input_dim = game_w * game_h * memory_size
		self.save_fig = save_fig_now
		self.progress_var = progress_var
		self.cancelled = False
		self.output = output
		self.steps_out = steps
		self.score_out= scored
		self.all_scores = score_tracker
		self.all_lived = step_tracker
		self.parent_instance = parent_instance
		self.game_tracker = game_tracker
		self.gui = gui
		self.gpu_acceleration = gpu_acceleration

		if gpu_acceleration:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		if m_type == "FCN":
			self.input_dim *= 3
			self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture)
			self.learning_model = networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture)
			self.encoding_type = "one_hot"
			


		elif m_type == "CNN":
			self.input_shape = (1,architecture[0].in_channels*memory_size,game_w,game_h)
			if self.gui:
				print(f"in shape is {self.input_shape}")
			self.target_model 	= networks.ConvolutionalNetwork(loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture,input_shape=self.input_shape,device=self.device)
			self.learning_model = networks.ConvolutionalNetwork(loss_fn=loss_fn,optimizer_fn=optimizer_fn,lr=lr,wd=wd,architecture=architecture,input_shape=self.input_shape,device=self.device)
			self.encoding_type = "6_channel"
			if self.gui:
				print(self.learning_model)

		total_params = sum(param.numel() for param in self.learning_model.model.parameters())

		#input(f"{m_type}/'
		# """'
		# '" model has {total_params} parameters")
		self.w = game_w	
		self.h = game_h
		self.target_model.to(self.device)
		self.learning_model.to(self.device)

		self.visible = visible
		self.movement_repr_tuples = [(0,-1),(0,1),(-1,0),(1,0)]
		self.gamma = gamma
		self.memory_size = memory_size
		self.loss_fn = loss_fn
		self.optimizer_fn = optimizer_fn
		self.lr = lr
		self.wd = wd
		self.epsilon = epsilon
		self.e_0 = self.epsilon
		self.architecture = architecture

		import pprint


	def train(self,episodes=1000,train_every=1000,replay_buffer=32768,sample_size=128,batch_size=32,epochs=10,early_stopping=True,transfer_models_every=2000,verbose=True,iters=3,picking=True):
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
						print(f"[Episode {str(e_i).rjust(len(str(episodes)))}/{int(episodes)}  -  {(100*e_i/episodes):.2f}% complete\t{(time.time()-t0):.2f}s\te: {self.epsilon:.2f}\thigh_score: {self.high_score}] lived_avg: {sum(lives[-1*train_every:])/len(lives[-1*train_every:]):.2f} score_avg: {sum(scores[-1*train_every:])/len(scores[-1*train_every:]):.2f}")
					self.steps_out.insert(0,f"{sum(lives[-1*train_every:])/len(lives[-1*train_every:]):.2f}")
					self.score_out.insert(0,f"{sum(scores[-1*train_every:])/len(scores[-1*train_every:]):.2f}")
					t0 = time.time()

					#Check score
					if self.high_score > self.best:
						self.best = self.high_score
					high_scores.append(self.high_score)
					self.high_score = 0


					best_sample = []

					if picking:
						blacklist = []
						indices = [i for i, item in enumerate(experiences) if item['r'] in [-2,2] ]
						quality = 100 * len(indices) / sample_size

						if verbose and e_i % 1024 == 0:
							print(f"quality of exps is {(100*quality / len(indices)+.1):.2f}%")
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
						if verbose and e_i % 1024 == 0:
							quality = sum(map(lambda x : int(x['r'] in [-2,2]),best_sample))
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
		fig.savefig(os.path.join("figs",f"{self.architecture}-{str(self.loss_fn).split('.')[-1][:-2]}-{str(self.optimizer_fn).split('.')[-1][:-2]}-ep{epochs}-lr{self.lr}-wd{self.wd}-bs{batch_size}-te{train_every}-rb{replay_buffer}-ss{sample_size}-p={picking}.png"),dpi=100)


		#Return the best score, high scores of all episode blocks, scores, and steps lived
		return scores,lived


	def train_concurrent(self,iters=1000,train_every=1024,pool_size=32768,sample_size=128,batch_size=32,epochs=10,early_stopping=True,transfer_models_every=2,verbose=False,picking=True,rewards={"die":-1,"eat":10,"step":-.01},max_steps=100,random_pick=False,blocker=256):
		
		#	Sliding window memory update 
		#	Instead of copying a new memory_pool list 
		#	upon overflow, simply replace the next window each time 

		memory_pool 	= []
		window_i 		= 0

		#	Keep track of models progress throughout the training 
		best_score 		= 0 

		#	Train 
		i = 0 
		while i < iters and not self.cancelled:
			#	Keep some performance variables 
			t0 				= time.time() 


			#	UPDATE EPSILON
			e 				= self.update_epsilon(i/(iters))	
			if self.gui:
				self.progress_var.set(i/iters)

			#	GET EXPERIENCES
			metrics, experiences, new_games = SnakeConcurrent.Snake(self.w,self.h,self.learning_model,simul_games=train_every,memory_size=self.memory_size,device=self.device,rewards=rewards,max_steps=max_steps).play_out_games(epsilon=e)



			#	UPDATE MEMORY POOL 
			#	replace every element of overflow with the next 
			# 	exp instead of copying the list every time 
			#	Is more efficient when memory_size >> len(experiences)
			for exp in experiences:
				if window_i < pool_size:
					memory_pool.append(None)
				memory_pool[window_i%pool_size] = exp 
				window_i += 1


			#Update metrics 
			for game_i,metr_dict in enumerate(metrics): 
				self.all_scores.append(metr_dict["highscore"])
				if self.all_scores[-1] > best_score:
					best_score = self.all_scores[-1]
					best_scorer = game_i
					if self.gui:
						self.output.insert(tk.END,f"  new hs: {best_score}\n")

				self.all_lived.append(metr_dict["lived_for"])
			
			self.game_tracker.append(new_games[game_i])

			#	UPDATE VERBOSE 
			if verbose:
				print(f"[Episode {str(i).rjust(15)}/{int(iters)} -  {(100*i/iters):.2f}% complete\t{(time.time()-t0):.2f}s\te: {e:.2f}\thigh_score: {best_score}\t] lived_avg: {(sum(self.all_lived[-1000:])/1000):.2f} score_avg: {(sum(self.all_scores[-1000:])/1000):.2f}")
			if self.gui:
				self.steps_out.set(f"{(sum(self.all_lived[-1000:])/1000):.2f}")
				self.score_out.set(f"{(sum(self.all_scores[-1000:])/1000):.2f}")
			
			# 	GET TRAINING SAMPLES
			#	AND TRAIN MODEL 
			if window_i > sample_size:

				if random_pick:
					training_set 	= random.sample(memory_pool,sample_size) 
					
				else:
					training_set 	= []
					training_ind	= []
					drop_rate = .8

					#Drop non-food examples with rate .75
					while len(training_set) < sample_size: 

						cur_i = random.randint(0,len(memory_pool)-1)
						while cur_i in training_ind:
							cur_i = random.randint(0,len(memory_pool)-1)

						if not memory_pool[cur_i]['r'] > 0:# or not memory_pool[cur_i]['r'] < .5: 

							if random.random() < drop_rate:
								continue 
							else:
								
								training_ind.append(cur_i)
								training_set.append(memory_pool[cur_i])
						else:
							if memory_pool[cur_i]['r'] < .5 and False:
								if random.random < drop_rate*.5:
									continue
							training_set.append(memory_pool[cur_i])
							training_ind.append(cur_i)

				qual 		= 100*sum([int(t['r'] > 0) for t in training_set]) / len(training_set)
				bad_set 	= random.sample(memory_pool,sample_size)
				bad_qual 	= f"{(100*sum([int(t['r'] > 0) for t in bad_set]) / len(bad_set)):.2f}"

				perc_str 	= f"{qual:.2f}%/{bad_qual}%".rjust(15)
				
				
				if verbose:
					print(f"[Quality\t{perc_str}  -  R_PICK: {'off' if random_pick else 'on'}\t\t\t\t\t\t]\n")
				self.train_on_experiences_better(training_set,epochs=epochs,batch_size=batch_size,early_stopping=False,verbose=verbose)

			#	UPDATE MODELS 
			if i/train_every % transfer_models_every == 0:
				self.transfer_models(transfer=True,verbose=verbose)
			
			i += train_every

			if self.gui:
				self.parent_instance.place_steps()
				self.parent_instance.place_scores()
		#	block_reduce lists 
		#plot up to 500

		if self.cancelled:
			print("exiting")
			return
		averager = int(len(self.all_scores)/blocker)
		
		while True:
			try:
				blocked_scores = numpy.average(numpy.array(self.all_scores).reshape(-1,averager),axis=1)
				blocked_lived = numpy.average(numpy.array(self.all_lived).reshape(-1,averager),axis=1)
				break 
			except ValueError:
				averager += 1



		x_scale = [averager*i for i in range(len(blocked_scores))] 
		
		graph_name = f"{self.name}_[{str(self.loss_fn).split('.')[-1][:-2]},{str(self.optimizer_fn).split('.')[-1][:-2]}@{self.lr}] x [{iters*train_every},{train_every}] mem size {pool_size} taking [{sample_size},{batch_size}]"

		if self.save_fig:
			plot_game(blocked_scores,blocked_lived,graph_name,x_scale)

		if self.gui:
			self.output.insert(tk.END,f"Completed Training\n\tHighScore:{best_score}\n\tSteps:{sum(self.all_lived[-100:])/100}")
		return blocked_scores,blocked_lived,best_score,x_scale,graph_name


	def train_on_experiences(self,big_set,epochs=100,batch_size=8,channels=6,early_stopping=True,verbose=False):
		
		for epoch_i in range(epochs):
			t0 = time.time()
			#Printing things
			if verbose and print(f"EPOCH {epoch_i}:\n\t",end='training['): pass
			next_percent = .02

			#Batch the sample set
			random.shuffle(big_set)
			n_batches = int(len(big_set)/batch_size)
			batches = [[big_set[i * n] for n in range(batch_size)] for i in range(n_batches)]
			init_states = torch.zeros(size=(n_batches,batch_size,channels,self.h,self.w),device=self.device)





			vals_target_adjusted_all = torch.zeros(size=(n_batches,batch_size,4),device=self.device)
			chosen_actions = [self.movement_repr_tuples[exp['a']] for exp in big_set]



			for i in range(int(len(big_set)/batch_size)):
				#Create the batch init states and batch 
				for j in range(batch_size):
					init_states[i,j] = big_set[i*j]["s`"]

			#Measrure losses and prepare for early stopping
			c_loss = 0
			prev_loss = 999999999999999

			#For each batch
			for i,batch in enumerate(batches):

				#Get a list (tensor) of all initial game states
				initial_states = init_states[i]

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
				vals_target_adjusted = torch.zeros((batch_size,4),device=self.device)

				#Apply Bellman
				for index,action in enumerate(chosen_action):

					# If state was terminal, use target reward
					if batch[index]['done']:
						target = batch[index]['r']

					# If not terminal, use Bellman Equation
					else:
						vect = batch[index]['s`']
						#   Q' <-       r          +    γ       *             max Q(s`) 
						target = batch[index]['r'] + self.gamma * torch.max(self.target_model(vect))

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


	def train_on_experiences_better(self,big_set,epochs=1,batch_size=8,early_stopping=True,verbose=False):
		
		#Telemetry 
		if verbose:
			print(f"TRAINING:")
			print(f"\tDataset:\n\t\t{'loss-fn'.ljust(12)}: {str(self.learning_model.loss).split('(')[0]}\n\t\t{'optimizer'.ljust(12)}: {str(self.learning_model.optimizer).split('(')[0]}\n\t\t{'size'.ljust(12)}: {len(big_set)}\n\t\t{'batch_size'.ljust(12)}: {batch_size}\n\t\t{'epochs'.ljust(12)}: {epochs}\n\t\t{'early-stop'.ljust(12)}: {early_stopping}\n")

		for epoch_i in range(epochs):
			#	Telemetry Vars 
			t0 			= time.time()
			t_gpu 		= 0
			num_equals 	= 53 
			printed 	= 0
			total_loss	= 0
			#	Telemetry
			if verbose:
				print(f"\tEPOCH: {epoch_i}\tPROGRESS- [",end='')
	
			#	Do one calc for all runs 
			num_batches = int(len(big_set) / batch_size)

			# Iterate through batches
			for batch_i in range(num_batches):
				
				#	Telemetry
				percent = batch_i / num_batches
				if verbose:
					while (printed / num_equals) < percent:
						print("=",end='',flush=True)
						printed+=1
				
				#Init final values for actual reward values 
				final_target_values = torch.ones(size=(batch_size,4),device=self.device,requires_grad=False)

				#Run all of batch through the network 
				batch_set 				= big_set[batch_i*batch_size:batch_i*batch_size+batch_size]
				exp_set 				= torch.stack([exp['s`'][0] for exp in batch_set])
				target_expected_values 	= torch.max(self.target_model.forward(exp_set),dim=1)[0]


				#One run of this for loop will be one batch run
				#Update the weights of the experience
				for item_i in range(batch_size):
					if self.cancelled:
						return
					#Pre calc some reused values
					i 					= item_i + (batch_i*batch_size)
					exp 				= big_set[i]

					#Calculate Bellman 
					final_target_values[item_i,exp["a"]] 	= exp["r"] + (exp['done'] * self.gamma * target_expected_values[item_i])



				#	BATCH GRADIENT DESCENT
				i_start 					= batch_i*batch_size
				i_end   					= i_start + batch_size

				#	Calculate Loss
				self.learning_model.optimizer.zero_grad()
				inputs 						= torch.stack([exp["s"][0] for exp in big_set[i_start:i_end]])
				t1 = time.time()
				this_batch 					= self.learning_model.forward(inputs)
				batch_loss 					= self.learning_model.loss(final_target_values,this_batch)
				total_loss 					+= batch_loss.item()

				#Back Propogate
				batch_loss.backward()
				self.learning_model.optimizer.step()
				t_gpu += time.time() - t1
			#	Telemetry
			if verbose :
				print(f"]\ttime: {(time.time()-t0):.2f}s\tt_gpu:{(t_gpu):.2f}\tloss: {(total_loss/num_batches):.6f}")
		if verbose:
			print("\n\n")


	def transfer_models(self,transfer=False,verbose=False):
		if transfer:
			if verbose:
				print("\ntransferring models\n\n")
			#Save the models

			#Check for dir 
			if not os.path.isdir(self.PATH):
				os.mkdir(self.PATH)
			torch.save(self.learning_model.state_dict(),os.path.join(self.PATH,f"{self.fname}_lm_state_dict"))
			#Load the learning model as the target model
			if self.m_type == "FCN":
				self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,lr=self.lr,wd=self.wd,architecture=self.architecture)
			elif self.m_type == "CNN":
				self.target_model = networks.ConvolutionalNetwork(loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,lr=self.lr,wd=self.wd,architecture=self.architecture,input_shape=self.input_shape)

			self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))
			self.target_model.to(self.device)

	@staticmethod
	def update_epsilon(percent):
		radical = -.0299573*100*percent -.916290 
		if percent > .90:
			return 0
		else:
			return pow(2.7182,radical)
	

