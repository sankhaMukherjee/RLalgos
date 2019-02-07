
import numpy as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim

class qNetworkDiscrete(nn.Module):

	def __init__(self, stateSize, actionSize, layers=[10, 5], activations=[F.tanh, F.tanh], batchNormalization = True, lr=0.01 ):
		'''This is a Q network with discrete actions
		
		This takes a state and returns a Q function for each action. Hence, the
		input is a state and the output is a set of Q values, one for each action
		in the action space. The action is assumed to be discrete. i.e. a ``1``
		when the particular action is to be desired.
		
		Parameters
		----------
		stateSize : {[type]}
			[description]
		actionSize : {[type]}
			[description]
		layers : {list}, optional
			[description] (the default is [10, 5], which [default_description])
		activations : {list}, optional
			[description] (the default is [F.tanh, F.tanh], which [default_description])
		batchNormalization : {bool}, optional
			[description] (the default is True, which [default_description])
		'''


		try:
			super(qNetworkDiscrete, self).__init__()
			self.stateSize           = stateSize
			self.actionSize          = actionSize
			self.layers              = layers
			self.activations         = activations
			self.batchNormalization  = batchNormalization

			# Generate the fullly connected layer functions
			self.fcLayers = []
			self.bns      = []

			oldN = stateSize
			for i, layer in enumerate(layers):
				self.fcLayers.append( nn.Linear(oldN, layer) )
				self.bns.append( nn.BatchNorm1d( num_features = layer ) )
				oldN = layer

			# ------------------------------------------------------
			# The final layer will only need to supply a quality
			# function. This is a single value for an action 
			# provided. Ideally, you would want to provide a 
			# OHE action sequence for most purposes ...
			# ------------------------------------------------------
			self.fcFinal = nn.Linear( oldN, actionSize )

			# we shall put this is eval mode and only use 
			# the trian mode when we need to train the 
			# mode
			self.optimizer = optim.Adam(
				self.parameters(), lr=lr)

		
		except Exception as e:
			print(f'Unable to generate the Q network ... : {e}')


		return

	def forward(self, x):
		'''forward function that is called during the forward pass
		
		This is the forward function that will be called during a 
		forward pass. It takes thee states and gives the Q value 
		correspondidng to each of the applied actions that are 
		associated with that state. 
		
		Parameters
		----------
		x : {tensor}
			This is a 2D tensor. 
		
		Returns
		-------
		tensor
			This represents the Q value of the function
		'''

		for i, (bn, fc, a) in enumerate(zip(self.bns, self.fcLayers, self.activations)):
			x = a(bn(fc(x)))

		x = self.fcFinal( x )

		return x

	def step(self, v1, v2):

		loss = F.mse_loss(v1, v2)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return
		