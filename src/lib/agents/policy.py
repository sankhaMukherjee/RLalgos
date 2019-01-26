import numpy as np

class epsGreedyPolicy:

	def __init__(self, agent, randomAgent):
		'''[summary]
		
		[description]
		
		Arguments:
			agent {[type]} -- [description]
			randomAgent {[type]} -- [description]
		'''

		self.agent       = agent
		self.randomAgent = randomAgent

		return

	def act(self, states, eps):
		'''[summary]
		
		[description]
		
		Arguments:
			states {[type]} -- [description]
			eps {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		'''

		actions = []
		for state in states:
			if np.random.rand() <= eps:
				actions.append( self.randomAgent( state ) )
			else:
				actions.append( self.agent( state ) )

		return actions