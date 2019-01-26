from logs import logDecorator as lD 
import json
import numpy as np

from unityagents import UnityEnvironment
from lib.envs    import envUnity
from lib.utils   import ReplayBuffer as RB

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testMemoryBuffer'


@lD.log( logBase + '.allTests' )
def allTests(logger):

	try:
		cfg  = json.load(open('../config/modules/tests.json'))['params']

		policy = lambda m: eval( cfg['MemoryBufferParams']['randomAction'] )
		memoryBuffer = RB.SimpleReplayBuffer(1000)

		print('Starting to generate memories ...')
		print('----------------------------------------')
		with envUnity.Env(cfg['MemoryBufferParams']['binaryFile'], showEnv=False) as env:

			for _ in range(10):
				print('[Generating Memories] ', end='', flush=True)
				allResults = env.episode(policy, maxSteps = 1000)
				memoryBuffer.appendAllAgentResults( allResults )
				
				print( 'Memory Buffer lengths: {}'.format( memoryBuffer.shape ) )
				

	except Exception as e:
		logger.error(f'Unable to finish Memory Buffer tests: {e}')


	return

