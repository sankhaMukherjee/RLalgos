
import numpy as np
import torch
import torch.nn            as nn
import torch.nn.functional as F

class SequentialCritic(nn.Module):

	def __init__(self, stateSize, actionSize, layers=[10, 5], activations=[F.tanh, F.tanh], mergeLayer = 0, batchNormalization = True ):
		'''[summary]
		
		[description]
		
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


		super(SequentialCritic, self).__init__()
		self.stateSize           = stateSize
		self.actionSize          = actionSize
		self.layers              = layers
		self.activations         = activations
		self.mergeLayer          = mergeLayer
		self.batchNormalization  = batchNormalization

		# Generate the fullly connected layer functions
		self.fcLayers = []
		self.bns      = []

		oldN = stateSize
		for i, layer in enumerate(layers):

			if i == mergeLayer:
				oldN += actionSize
			self.fcLayers.append( nn.Linear(oldN, layer) )
			self.bns.append( nn.BatchNorm1d( num_features = layer ) )
			oldN = layer

		# ------------------------------------------------------
		# The final layer will only need to supply a quality
		# function. This is a single value and so the output
		# is just this one value. This will not need activation
		# ------------------------------------------------------
		self.fcFinal = nn.Linear( oldN, 1 )

		return

	def forward(self, x, action):

		for i, (bn, fc, a) in enumerate(zip(self.bns, self.fcLayers, self.activations)):
			if i == self.mergeLayer:
				x = torch.cat((x, action), dim=1)
			x = a(bn(fc(x)))

		x = self.fcFinal( x )

		return x
		