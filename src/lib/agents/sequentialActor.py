
import numpy as np
import torch
import torch.nn            as nn
import torch.nn.functional as F

class SequentialDiscreteActor(nn.Module):

	def __init__(self, stateSize, numActions, layers=[10, 5], activations=[F.tanh, F.tanh], batchNormalization = True ):
		'''[summary]
		
		[description]
		
		Arguments:
			stateSize {[type]} -- [description]
			numActions {[type]} -- [description]
		
		Keyword Arguments:
			layers {list} -- [description] (default: {[10, 5]})
			activations {list} -- [description] (default: {[F.tanh, F.tanh]})
			batchNormalization {bool} -- [description] (default: {True})
		'''


		super(SequentialDiscreteActor, self).__init__()
		self.stateSize           = stateSize
		self.numActions          = numActions
		self.layers              = layers
		self.activations         = activations
		self.batchNormalization  = batchNormalization

		# Generate the fullly connected layer functions
		self.fcLayers = []
		self.bns      = []

		oldN = stateSize
		for layer in layers:
			self.fcLayers.append( nn.Linear(oldN, layer) )
			self.bns.append( nn.BatchNorm1d( num_features = layer ) )
			oldN = layer

		# ------------------------------------------------------
		# There is one final layer that needs to be added
		# as the output layer. This will have a sooftmax
		# activation that will allow a one of many activation
		# approach
		# ------------------------------------------------------
		self.fcFinal = nn.Linear( oldN, numActions )

		return

	def forward(self, x):

		for bn, fc, a in zip(self.bns, self.fcLayers, self.activations):
			x = a(bn(fc(x)))

		x = F.softmax(x)

		return x
		