import networks 
import torch 
import os 
import time 
import SnakeConcurrentIMG
import random 
from matplotlib import pyplot as plt 
import numpy 
import sys 
import tkinter as tk
from snakeAI import SnakeGame
from telemetry import plot_game
import copy

class Trainer:

	def __init__(	self,game_w,game_h,
					visible=True,
					loading=True,
					PATH="models",
					memory_size=4,
					loss_fn=torch.nn.MSELoss,
					optimizer_fn=torch.optim.Adam,
					kwargs={"lr":1e-5},
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
					best_score=[],
					best_game=[],
					game_tracker=[],
					gui=False,
					instance=False,
					channels=3):



		#Set file handling vars 
		self.PATH 				= PATH
		self.fname 				= fname
		self.name 				= name
		self.save_fig 			= save_fig_now

		#Set model vars  
		self.m_type 			= m_type
		self.input_dim 			= game_w * game_h * memory_size
		self.progress_var 		= progress_var
		self.gpu_acceleration 	= gpu_acceleration
		self.movement_repr_tuples = [(0,-1),(0,1),(-1,0),(1,0)]
		self.loss_fn = loss_fn
		self.optimizer_fn 		= optimizer_fn
		self.architecture 		= architecture

		#Set runtime vars 
		self.cancelled 			= False
		self.w 					= game_w	
		self.h 					= game_h
		self.visible 			= visible

		#Set telemetry vars 
		self.steps_out 			= steps
		self.score_out			= scored
		self.all_scores 		= score_tracker
		self.all_lived 			= step_tracker
		self.output 			= output
		self.game_tracker 		= game_tracker
		self.gui 				= gui
		self.best_score			= best_score
		self.best_game			= best_game
		self.instance 	 		= instance 

		#Set training vars 
		self.gamma 				= gamma
		self.memory_size 		= memory_size
		self.epsilon 			= epsilon
		self.e_0 				= self.epsilon
		self.kwargs				= kwargs
		#Enable cuda acceleration if specified 
		self.device 			= torch.device('cuda') if gpu_acceleration else torch.device('cpu')




		#Generate models for the learner agent 
		if m_type == "FCN":
			self.input_dim *= 3
			self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,architecture=architecture)
			self.learning_model = networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,architecture=architecture)
			self.encoding_type = "one_hot"
		elif m_type == "CNN":
			self.input_shape = (1,channels*memory_size,game_w,game_h)
			self.target_model 	= networks.IMG_NET(loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,input_shape=self.input_shape,device=self.device)
			
			if self.gui:
				self.output.insert(tk.END,f"Generated training model\n\t{sum([p.numel() for p in self.target_model.model.parameters()])} params")
			self.learning_model = networks.IMG_NET(loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,input_shape=self.input_shape,device=self.device)
			self.encoding_type = "6_channel"
		self.target_model.to(self.device)
		self.learning_model.to(self.device)

		networks.init_weights(self.learning_model)
		networks.init_weights(self.target_model)

		#Set optimizer for conv filters 
		torch.backends.cudnn.benchmark = True


	def train_concurrent(self,iters=1000,train_every=1024,pool_size=32768,sample_size=128,batch_size=32,epochs=10,transfer_models_every=2,verbose=False,rewards={"die":-3	 ,"eat":5,"step":-.01},max_steps=100,random_pick=True,blocker=256,drop_rate=.25):
		
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
			metrics, experiences, new_games = SnakeConcurrentIMG.Snake(self.w,self.h,self.learning_model,simul_games=train_every,memory_size=self.memory_size,device=self.device,rewards=rewards,max_steps=max_steps).play_out_games(epsilon=e)



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
			#Find top scorer this round
			round_best_scorer= 0
			round_best_score = 0

			for game_i,metr_dict in enumerate(metrics): 
				
				#Update large telemetry 
				score = metr_dict["highscore"]
				self.all_scores.append(score)
				self.all_lived.append(metr_dict["lived_for"])
				
				#Check best scorer
				if score > round_best_score:
					round_best_score = score 
					round_best_scorer = game_i					
			
			#Save best game for telemetry 
			self.game_tracker.append(new_games[round_best_scorer])

			#Check for best score ever
			if round_best_score >= best_score:
				if self.gui:
					self.instance.best_score = best_score
					self.instance.best_game = copy.deepcopy(new_games[round_best_scorer])
					
					if round_best_score > best_score:
						best_score = round_best_score
						self.output.insert(tk.END,f"  new hs: {best_score}\n")
			

			#	UPDATE VERBOSE 
			if verbose:
				print(f"[Episode {str(i).rjust(15)}/{int(iters)} -  {(100*i/iters):.2f}% complete\t{(time.time()-t0):.2f}s\te: {e:.2f}\thigh_score: {best_score}\t] lived_avg: {(sum(self.all_lived[-100:])/100):.2f} score_avg: {(sum(self.all_scores[-100:])/100):.2f}")
			if self.gui:
				self.instance.var_step.set(f"{(sum(self.all_lived[-100:])/100):.2f}")
				self.instance.var_score.set(f"{(sum(self.all_scores[-100:])/100):.2f}")
			
			# 	GET TRAINING SAMPLES
			#	AND TRAIN MODEL 
			if window_i > sample_size:
				
				#PICK RANDOMLY 
				if random_pick:
					training_set 	= random.sample(memory_pool,sample_size) 
				
				#PICK SELECTIVELY
				else:
					training_set 	= []
					training_ind	= []
					drop_rate = .2

					while len(training_set) < sample_size: 

						cur_i = random.randint(0,len(memory_pool)-1)						#Pick new random index 
						while cur_i in training_ind:
							cur_i = random.randint(0,len(memory_pool)-1)

						#Drop non-scoring experiences with odds: 'drop_rate'
						is_non_scoring 				= abs(memory_pool[cur_i]['r']) < .1
						if is_non_scoring and random.random() < drop_rate:
							continue
								
						else:
							training_set.append(memory_pool[cur_i])
							training_ind.append(cur_i)

				qual 		= 100*sum([int(t['r'] > 1) + int(t['r'] < -1) for t in training_set]) / len(training_set)
				bad_set 	= random.sample(memory_pool,sample_size)
				bad_qual 	= f"{100*sum([int(t['r'] > 1) + int(t['r'] < -1) for t in bad_set]) / len(bad_set):.2f}"

				perc_str 	= f"{qual:.2f}%/{bad_qual}%".rjust(15)
				
				
				if verbose:
					print(f"[Quality\t{perc_str}  -  R_PICK: {'off' if random_pick else 'on'}\t\t\t\t\t\t]\n")
				self.train_on_experiences_better(training_set,epochs=epochs,batch_size=batch_size,early_stopping=False,verbose=verbose)

				if self.instance.cancel_var:
					return 
			#	UPDATE MODELS 
			if i/train_every % transfer_models_every == 0:
				self.transfer_models(transfer=True,verbose=verbose)
			
			i += train_every

			if self.gui:
				
				self.instance.training_epoch_finished = True
		#	block_reduce lists 
		#plot up to 500

		
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


	def train_on_experiences_better(self,big_set,epochs=1,batch_size=8,early_stopping=True,verbose=False):
		
		#Telemetry 
		if verbose:
			print(f"TRAINING:")
			print(f"\tDataset:\n\t\t{'loss-fn'.ljust(12)}: {str(self.learning_model.loss).split('(')[0]}\n\t\t{'optimizer'.ljust(12)}: {str(self.learning_model.optimizer).split('(')[0]}\n\t\t{'size'.ljust(12)}: {len(big_set)}\n\t\t{'batch_size'.ljust(12)}: {batch_size}\n\t\t{'epochs'.ljust(12)}: {epochs}\n\t\t{'early-stop'.ljust(12)}: {early_stopping}\n")

		for epoch_i in range(epochs):
			if self.instance.cancel_var:
				return
			#	Telemetry Vars 
			t0 			= time.time()
			t_gpu 		= 0
			num_equals 	= 45 
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
				final_target_values = torch.zeros(size=(batch_size,4),device=self.device,requires_grad=False)

				#Run all of batch through the network 
				batch_set 				= big_set[batch_i*batch_size:batch_i*batch_size+batch_size]
				#exp_set 				= torch.stack([exp['s`'][0] for exp in batch_set])
				exp_set 				= torch.stack([exp['s`'][0] for exp in batch_set]).type(torch.float)
				with torch.no_grad():
					target_expected_values 	= torch.max(self.target_model.forward(exp_set),dim=1)[0]
				#input(f"initial vals\n{final_target_values}")

				#One run of this for loop will be one batch run
				#Update the weights of the experience
				for item_i in range(batch_size):

					if self.cancelled:
						return

					#Pre calc some reused values
					exp 				= batch_set[item_i]

					#print(f"EXP is {exp['s']}")
					#print(f"chosen action was {exp['a']}")
					#print(f"resulted in next state {exp['s`']}")
					#print(f"done val: {exp['done']}")
					#Calculate Bellman 
					final_target_values[item_i,exp["a"]] 	= exp["r"] + (exp['done'] * self.gamma * target_expected_values[item_i])
					#print(f"final value is {final_target_values[item_i,exp['a']]}")
					#input()

				#input(f"updated trgVals\n{final_target_values}")

				#	BATCH GRADIENT DESCENT
				i_start 					= batch_i*batch_size
				i_end   					= i_start + batch_size

				#	Calculate Loss
				self.learning_model.optimizer.zero_grad()
				#inputs 						= torch.stack([exp["s"][0] for exp in big_set[i_start:i_end]])
				inputs 						= torch.stack([exp["s"][0] for exp in big_set[i_start:i_end]]).type(torch.float)
				t1 = time.time()
				this_batch 					= self.learning_model.forward(inputs)
				#input(f"This batch\n{this_batch}")
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
				self.target_model = networks.IMG_NET(loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,kwargs=self.kwargs,input_shape=self.input_shape,device=self.device)

			self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))
			self.target_model.to(self.device)

	@staticmethod
	def update_epsilon(percent):
		radical = -.4299573*100*percent -1.2116290 
		if percent > .50:
			return 0
		else:
			return pow(2.7182,radical)
	

