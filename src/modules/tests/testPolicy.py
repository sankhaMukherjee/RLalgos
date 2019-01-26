from logs import logDecorator as lD 
import json
import numpy as np

from unityagents import UnityEnvironment
from lib.envs    import envUnity
from lib.utils   import ReplayBuffer as RB
from lib.agents  import policy, randomAgent as rA

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testPolicy'


@lD.log( logBase + '.allTests' )
def allTests(logger):

	try:
		cfg  = json.load(open('../config/modules/tests.json'))['params']

		agent  = rA.randomDiscreteAgent((37,), 4)
		rAgent = rA.randomDiscreteAgent((37,), 4)
		egPolicy = policy.epsGreedyPolicy( agent, rAgent )

		# At any point this policy can be changed ...
		policy1 = lambda states: egPolicy.act( states , 0.1)
		memoryBuffer = RB.SimpleReplayBuffer(1000)

		print('Starting to generate memories ...')
		print('----------------------------------------')
		with envUnity.Env(cfg['policyParams']['binaryFile'], showEnv=False) as env:

			for _ in range(10):
				print('[Generating Memories] ', end='', flush=True)
				allResults = env.episode(policy1, maxSteps = 1000)
				memoryBuffer.appendAllAgentResults( allResults )
				
				print( 'Memory Buffer lengths: {}'.format( memoryBuffer.shape ) )
				

	except Exception as e:
		logger.error(f'Unable to finish Memory Buffer tests: {e}')


	return

