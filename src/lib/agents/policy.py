

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

		return actions