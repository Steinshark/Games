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
from utilities import reduce_arr

class Trainer:

	def __init__(	self,
	      			game_w,
	      			game_h,
					visible=True,
					loading=True,
					PATH="models",
					min_thresh=.03,
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
					best_score=0,
					best_game=[],
					game_tracker=[],
					gui=False,
					instance=None,
					channels=3,
					display_img=False,
					lr_threshs=None,
					dropout_p=.1):



		#Set file handling vars 
		self.PATH 				= PATH
		self.fname 				= fname
		self.name 				= name
		self.save_fig 			= save_fig_now

		#Set model vars  
		self.m_type 			= m_type
		self.input_dim 			= game_w * game_h
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
		self.best_score			= 0
		self.best_game			= best_game
		self.instance 	 		= instance 
		self.base_threshs		= [(-1,.0001),(1024+256,.00001),(1024+512+256,3e-6),(2048,1e-6),(4096,5e-7),(4096+2048,2.5e-7),(8192,1e-7),(8192*2,1e-8)] if not lr_threshs else lr_threshs
		self.display_img		= display_img

		#Set training vars 
		self.gamma 				= gamma
		self.min_thresh 		= min_thresh
		self.epsilon 			= epsilon
		self.e_0 				= self.epsilon
		self.kwargs				= kwargs
		self.softmax 			= True
		self.dropout_p			= dropout_p
		#Enable cuda acceleration if specified 
		self.device 			= torch.device('cuda') if gpu_acceleration else torch.device('cpu')




		#Generate models for the learner agent 
		if m_type == "FCN":
			self.input_dim *= 3
			self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,architecture=architecture)
			self.learning_model = networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,architecture=architecture)
			self.encoding_type = "one_hot"
		elif m_type == "CNN":
			self.input_shape = (1,channels,game_w,game_h)
			self.target_model 	= networks.IMG_NET(loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,input_shape=self.input_shape,device=self.device,dropout_p=self.dropout_p)
			
			if self.gui:
				self.output.insert(tk.END,f"Generated training model\n\t{sum([p.numel() for p in self.target_model.model.parameters()])} params")
			self.learning_model = networks.IMG_NET(loss_fn=loss_fn,optimizer_fn=optimizer_fn,kwargs=kwargs,input_shape=self.input_shape,device=self.device,dropout_p=self.dropout_p)
			self.encoding_type = "6_channel"
		self.target_model.to(self.device)
		self.learning_model.to(self.device)

		networks.init_weights(self.learning_model)
		networks.init_weights(self.target_model)

		#Set optimizer for conv filters 
		torch.backends.cudnn.benchmark = True


	def train_concurrent(	self,
		      				iters=1000,
							train_every=1024,
							pool_size=32768,
							sample_size=128,
							batch_size=32,
							epochs=10,
							transfer_models_every=2,
							verbose=False,
							rewards={"die":-3,"eat":5,"step":-.01},
							max_steps=100,
							random_pick=True,
							drop_rate=.25,
							x_scale=100,
							timeout="inf"):
		
		#	Sliding window memory update 
		#	Instead of copying a new memory_pool list 
		#	upon overflow, simply replace the next window each time 
		self.tstart 	= time.time()
		self.x_scale 	= x_scale
		memory_pool 	= []
		window_i 		= 0
		display_img		= False
		self.pending_graph	= False


		threshs			= copy.deepcopy(self.base_threshs)
		stop_thresh 	= False 
		#	Keep track of models progress throughout the training 
		best_score 		= 0 

		#	Train 
		i = 0 
		while i < iters and not self.cancelled:
			
			#Check no timeout 
			if not timeout in ["none","inf"] and (time.time() - self.tstart) > timeout:
				return self.cleanup()

			#	Keep some performance variables 
			t0 				= time.time() 
			
			# 	LR Scheduler
			if not stop_thresh and i > threshs[0][0]:
				new_lr 	= threshs[0][1]
				self.learning_model.optimizer.param_groups[0]['lr']	= new_lr
				if not len(threshs) == 1:
					threshs = threshs[1:] 
				else:
					stop_thresh	= True 
				
				if self.gui:
					self.output.insert(tk.END,f"\tlr updated to: {new_lr}\n")
					
			#	UPDATE EPSILON
			e 				= self.update_epsilon(i/(iters))	
			if self.gui:
				self.progress_var.set(i/iters)
				display_img			= self.instance.settings['dspl'].get()
				
		
			#	GET EXPERIENCES
			metrics, experiences, new_games  = SnakeConcurrentIMG.Snake(self.w,self.h,self.learning_model,simul_games=train_every,device=self.device,rewards=rewards,max_steps=max_steps,min_thresh=self.min_thresh).play_out_games(epsilon=e,display_img=display_img)

			#	UPDATE MEMORY POOL 
			#	replace every element of overflow with the next 
			# 	exp instead of copying the list every time 
			#	Is more efficient when memory_size >> len(experiences)
			for exp in experiences:
				if window_i < pool_size:
					memory_pool.append(None)
				memory_pool[window_i%pool_size] = exp 
				window_i += 1


			#	UPDATE METRICS
			#	Add average metrics to telemetry 
			self.all_scores.append(sum([game['highscore'] for game in metrics])/len(metrics))
			self.all_lived.append(sum([game['lived_for'] for game in metrics])/len(metrics))

			#	Find best game
			scores 				= [game['highscore'] for game in metrics]
			round_top_score 	= max(scores)
			round_top_game 		= new_games[scores.index(round_top_score)]
			self.game_tracker.append(round_top_game)

			#	Update local and possibly gui games
			if round_top_score >= self.best_score:
				self.best_score 	= round_top_score

				if self.gui:
					self.instance.best_game 	= copy.deepcopy(round_top_game)
					if round_top_score > self.instance.best_score:
						self.instance.best_score	= round_top_score
						self.output.insert(tk.END,f"\tnew hs: {self.best_score}\n")
			

			#	UPDATE VERBOSE 
			if verbose:
				print(f"[Episode {str(i).rjust(15)}/{int(iters)} -  {(100*i/iters):.2f}% complete\t{(time.time()-t0):.2f}s\te: {e:.2f}\thigh_score: {self.best_score}\t] lived_avg: {(sum(self.all_lived[-100:])/100):.2f} score_avg: {(sum(self.all_scores[-100:])/100):.2f}")
			
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

				qual 		= 100*sum([int(t['r'] >= 1) + int(t['r'] <= -1) for t in training_set]) / len(training_set)
				bad_set 	= random.sample(memory_pool,sample_size)
				bad_qual 	= f"{100*sum([int(t['r'] >= 1) + int(t['r'] <= -1) for t in memory_pool]) / len(memory_pool):.2f}"

				perc_str 	= f"{qual:.2f}%/{bad_qual}%".rjust(15)
				
				
				if verbose:
					print(f"[Quality\t{perc_str}  -  R_PICK: {'off' if random_pick else 'on'}\t\t\t\t\t\t]\n")
				self.train_on_experiences(training_set,epochs=epochs,batch_size=batch_size,early_stopping=False,verbose=verbose)

				if self.gui and self.instance.cancel_var:
					self.output.insert(tk.END,f"CANCELLING\n")
					return 
			
			#	UPDATE MODELS 
			if i/train_every % transfer_models_every == 0:
				self.transfer_models(transfer=True,verbose=verbose)
			
			i += train_every

			if self.gui:
				self.instance.training_epoch_finished = True

		#plot up to 500
		return self.cleanup()

		
	def cleanup(self):
		blocked_scores		= reduce_arr(self.all_scores,self.x_scale)
		blocked_lived 		= reduce_arr(self.all_lived,self.x_scale)
		graph_name = f"{self.name}_[{str(self.loss_fn).split('.')[-1][:-2]},{str(self.optimizer_fn).split('.')[-1][:-2]}@{self.kwargs['lr']}]]]"

		if self.save_fig:
			plot_game(blocked_scores,blocked_lived,graph_name)

		if self.gui:
			self.output.insert(tk.END,f"Completed Training\n\tHighScore:{self.best_score}\n\tSteps:{sum(self.all_lived[-1000:])/1000}")
		return blocked_scores,blocked_lived,self.best_score,graph_name


	def train_on_experiences(self,big_set,epochs=1,batch_size=8,early_stopping=True,verbose=False):
		
		#Telemetry 
		if verbose:
			print(f"TRAINING:")
			print(f"\tDataset:\n\t\t{'loss-fn'.ljust(12)}: {str(self.learning_model.loss).split('(')[0]}\n\t\t{'optimizer'.ljust(12)}: {str(self.learning_model.optimizer).split('(')[0]}\n\t\t{'size'.ljust(12)}: {len(big_set)}\n\t\t{'batch_size'.ljust(12)}: {batch_size}\n\t\t{'epochs'.ljust(12)}: {epochs}\n\t\t{'lr'.ljust(12)}: {self.learning_model.optimizer.param_groups[0]['lr']:.8f}\n")

		for epoch_i in range(epochs):
			if self.gui and self.instance.cancel_var:
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

				i_start 					= batch_i*batch_size
				i_end   					= i_start + batch_size
				
				#	Telemetry
				percent = batch_i / num_batches
				if verbose:
					while (printed / num_equals) < percent:
						print("=",end='',flush=True)
						printed+=1
				

				#BELLMAN UPDATE 
				self.learning_model.optimizer.zero_grad()

				#Gather batch experiences
				batch_set 							= big_set[i_start:i_end]

				init_states 						= torch.stack([exp['s'][0]  for exp in batch_set]).type(torch.float)
				action 								= [exp['a'] for exp in batch_set]
				next_states							= torch.stack([exp['s`'][0] for exp in batch_set]).type(torch.float)
				rewards 							= [exp['r']  for exp in batch_set]
				done								= [exp['done'] for exp in batch_set]
				
				#Calc final targets 
				initial_target_predictions 			= self.learning_model.forward(init_states)
				final_target_values 				= initial_target_predictions.clone().detach()
				
				#Get max from s`
				with torch.no_grad():
					stepped_target_predictions 		= self.target_model.forward(next_states)
					best_predictions 				= torch.max(stepped_target_predictions,dim=1)[0]

				#Update init values 
				for i,val in enumerate(best_predictions):
					chosen_action						= action[i]
					final_target_values[i,chosen_action]	= rewards[i] + (done[i] * self.gamma * val)

				#	Calculate Loss
				t1 							= time.time()
				batch_loss 					= self.learning_model.loss(initial_target_predictions,final_target_values)
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

			# prev_state_dict 			= self.learning_model.state_dict()
			# self.target_model 			= networks.IMG_NET(loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,kwargs=self.kwargs,input_shape=self.input_shape,device=self.device,dropout_p=self.dropout_p)
			# self.target_model.load_state_dict(self.learning_model.state_dict())
			# return 
		
			torch.save(self.learning_model.state_dict(),os.path.join(self.PATH,f"{self.fname}_lm_state_dict"))
			#Load the learning model as the target model
			if self.m_type == "FCN":
				self.target_model 	= networks.FullyConnectedNetwork(self.input_dim,4,loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,lr=self.lr,wd=self.wd,architecture=self.architecture)
			elif self.m_type == "CNN":
				self.target_model = networks.IMG_NET(loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,kwargs=self.kwargs,input_shape=self.input_shape,device=self.device)
				self.target_model = networks.IMG_NET(loss_fn=self.loss_fn,optimizer_fn=self.optimizer_fn,kwargs=self.kwargs,input_shape=self.input_shape,device=self.device,dropout_p=self.dropout_p)


			self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))
			self.target_model.to(self.device)

	@staticmethod
	def update_epsilon(percent):
		radical = -.4299573*100*percent -1.2116290 
		if percent > .50:
			return 0
		else:
			return pow(2.7182,radical)
	

