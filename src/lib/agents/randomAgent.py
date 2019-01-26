
import numpy as np

class randomDiscreteAgent():

	def __init__(self, stateShape, numActions):
		'''a random discrete agent
		
		A discrete action space is one where the agent will return
		an integer which will represent one of n actions. This agent
		will return a random action independent of the state the 
		environment is in. This will be roughly uniformly distributed.
		
		Arguments:

			stateShape {tuple} -- Tuple of integers that will describe
				the dimensions of the state space

			numActions {integer} -- An action that the agent will do.
		
		'''

		self.stateShape = stateShape 
		self.numActions = numActions

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

	def save(self, location=None):

		return

	def restore(self, location=None):

		return
