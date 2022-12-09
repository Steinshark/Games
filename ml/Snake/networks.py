from platform import architecture
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import random

class FullyConnectedNetwork(nn.Module):
	def __init__(self,input_size,output_size,loss_fn=None,optimizer_fn=None,lr=1e-6,wd=1e-6,architecture=[512,32,16]):
		super(FullyConnectedNetwork,self).__init__()

		self.model = nn.Sequential(nn.Linear(input_size,architecture[0]))
		self.model.append(nn.LeakyReLU(.5))

		for i,size in enumerate(architecture[:-1]):

			self.model.append(nn.Linear(size,architecture[i+1]))
			self.model.append(nn.LeakyReLU(.5))
		self.model.append(nn.Linear(architecture[-1],output_size))
		self.model.append(nn.Softmax(dim=0))
		self.optimizer = optimizer_fn(self.model.parameters(),lr=lr,weight_decay=wd)
		self.loss = loss_fn()

	def train(self,x_input,y_actual,epochs=1000,verbose=False,show_steps=10,batch_size="online",show_graph=False):
		memory = 3
		prev_loss = [100000000 for x in range(memory)]
		losses = []
		if type(batch_size) is str:
			batch_size = len(y_actual)

		if verbose:
			print(f"Training on dataset shape:\t f{x_input.shape} -> {y_actual.shape}")
			print(f"batching size:\t{batch_size}")

		#Create the learning batches
		dataset = torch.utils.data.TensorDataset(x_input,y_actual)
		dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)


		for i in range(epochs):
			#Track loss avg in batch
			avg_loss = 0

			for batch_i, (x,y) in enumerate(dataloader):

				#Find the predicted values
				batch_prediction = self.forward(x)
				#Calculate loss
				loss = self.loss(batch_prediction,y)
				avg_loss += loss
				#Perform grad descent
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			avg_loss = avg_loss / batch_i 			# Add losses to metric's list
			losses.append(avg_loss.cpu().detach().numpy())

			#Check for rising error
			if not False in [prev_loss[x] > prev_loss[x+1] for x in range(len(prev_loss)-1)]:
				print(f"broke on epoch {i}")
				break
			else:
				prev_loss = [avg_loss] + [prev_loss[x+1] for x in range(len(prev_loss)-1)]

			#Check for verbosity
			if verbose and i % show_steps == 0:
				print(f"loss on epoch {i}:\t{loss}")

		if show_graph:
			plt.plot(losses)
			plt.show()


	def forward(self,x_list):
		return self.model(x_list)
		#	y_predicted.append(y_pred.cpu().detach().numpy())


class ConvolutionalNetwork(nn.Module):
	
	def __init__(self,loss_fn=None,optimizer_fn=None,lr=1e-6,wd=1e-6,architecture=[[3,2,5,3,2]],input_shape=(1,3,30,20)):
		super(ConvolutionalNetwork,self).__init__()
		self.model = []
		switched = False 
		self.input_shape = input_shape

		self.activation = {	"relu" : nn.ReLU,
							"sigmoid" : nn.Sigmoid}

		for i,layer in enumerate(architecture):
			if len(layer) == 3:
				in_c,out_c,kernel = layer[0],layer[1],layer[2]
				self.model.append(nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=kernel,padding=2))
				self.model.append(self.activation['relu']())
				
			else:
				in_size, out_size = layer[0],layer[1]
				if not switched:
					self.model.append(nn.Flatten(1))
					switched = True 
				self.model.append(nn.Linear(in_size,out_size))
				if not i == len(architecture)-1 :
					self.model.append(self.activation['relu']())

		#self.model.append(nn.Softmax(1))
		o_d = OrderedDict({str(i) : n for i,n in enumerate(self.model)})
		self.model = nn.Sequential(o_d)
		self.loss = loss_fn()
		self.optimizer = optimizer_fn(self.model.parameters(),lr=lr)
		
	def train(self,x_input,y_actual,epochs=10,in_shape=(1,6,10,10)):

		#Run epochs
		for i in range(epochs):

			#Predict on x : M(x) -> y
			y_pred = self.model(x_input)
			#Find loss  = y_actual - y
			loss = self.loss_function(y_pred,y_actual)
			print(f"epoch {i}:\nloss = {loss}")

			#Update network
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

	def forward(self,x):
		#if len(x.shape) == 3:
			#input(f"received input shape: {x.shape}")
			#x = torch.reshape(x,self.input_shape)
		
		return self.model(x)



class ConvNet(nn.Module):
	def __init__(self,architecture,loss_fn=nn.MSELoss,optimizer=torch.optim.SGD,lr=.005):
		self.layers = {}

		for l in architecture:
			#Add Conv2d layers
			if len(l) == 3:
				in_c,out_c,kernel_size = l[0],l[1],l[2]
				self.layers.append(len(self.layers),nn.Conv2d(in_c,out_c,kernel_size))
				self.layers.append(nn.ReLU())
			#Add Linear layers
			elif len(l) == 2:
				in_dim,out_dim = l[0],l[1]
				self.layers[len(self.layers) : nn.Linear(in_dim,out_dim)]
				if not l == architecture[-1]:
					self.layers.append(nn.ReLU())
		
		self.loss = loss_fn()


if __name__ == "__main__":
	function = lambda x : math.sin(x*.01) + 4
	x_fun = lambda x : [x, x**2, 1 / (x+.00000001), math.sin(x * .01)]
	x_train = torch.tensor([[x] for x in range(2000) if random.randint(0,100) < 80],dtype=torch.float)
	x_train1 =  torch.tensor([x_fun(x) for x in range(2000) if random.randint(0,100) < 80],dtype=torch.float)
	y_train = torch.tensor([[function(x[0])] for x in x_train1],dtype=torch.float)

	#print(x_train.shape)
	#print(y_train.shape)
	#plt.scatter(x_train,y_train)
	#plt.show()
	print("Prelim dataset")

	model = FullyConnectedNetwork(len(x_train1[0]))
	model.train(x_train1,y_train)



	x_pred = torch.tensor([[x] for x in range(2000) if random.randint(0,100) < 20],dtype=torch.float)
	x_pred1 = torch.tensor([x_fun(x) for x in range(2000) if random.randint(0,100) < 20],dtype=torch.float)

	y_actual = torch.tensor([[function(x[0])] for x in x_pred1],dtype=torch.float)


	y_pred = model.forward(x_pred1).cpu().detach().numpy()

	plt.scatter([i for i in range(len(x_pred1))],y_actual)
	plt.scatter([i for i in range(len(x_pred1))],y_pred)
	plt.show()
	print("model output")
