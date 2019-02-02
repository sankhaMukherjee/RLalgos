
import numpy as np
import torch
import torch.nn            as nn
import torch.nn.functional as F

class fullyConnectedDiscreteActor(nn.Module):

	def __init__(self, stateShape, numActions, hiddenSizes=[24, 48]):
		'''a fully-connected discrete actor
		
		A discrete action space is one where the actor will return
		an integer which will represent one of n actions. 
		
		Arguments:

			stateShape {tuple} -- Tuple of integers that will describe
				the dimensions of the state space

			numActions {integer} -- An action that the agent will do.
		
		'''

		super(fullyConnectedDiscreteActor, self).__init__()
		self.stateShape = stateShape 
		self.numActions = numActions

		self.fcns = []
		self.bns  = []

		allSizes = hiddenSizes + [numActions]
		prevS    = stateShape[-1]

		for s in allSizes:
			self.fcns.append(nn.Linear( prevS, s ))
			self.bns.append(nn.BatchNorm1d( num_features = s ))
			prevS = s

		return

	def act(self, state):
		'''return an action based on the state
		
		[description]
		
		Arguments:
			state {nd-array} -- nd-array as described by the state
				shape described in the ``__init__`` function. 
		
		Returns:
			integer -- integer between 0 and the number of actions
				available.
		'''

		result = np.random.randint(0, self.numActions)

		return result

