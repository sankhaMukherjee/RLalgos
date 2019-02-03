from logs import logDecorator as lD 
import json
import numpy as np

import torch
from torch.nn import functional as F

from lib.agents  import sequentialCritic as sC

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testPolicy'

@lD.log( logBase + '.allTests' )
def allTests(logger):

	try:

		sequence = sC.SequentialCritic(
			stateSize = 10, actionSize = 1, 
			layers                  = [20, 10, 3, 4], 
			activations             = [F.tanh, F.tanh, F.tanh, F.tanh], 
			mergeLayer              = 3, 
			batchNormalization      = True)

		inp = np.random.rand(30, 10).astype( np.float32 ) * 10
		actions = np.random.rand(30, 1).astype( np.float32 ) * 10
		out = sequence( torch.as_tensor(inp), torch.as_tensor(actions) )


		print( out )

	except Exception as e:
		logger.error(f'Unable to finish Critic tests: {e}')

	return

