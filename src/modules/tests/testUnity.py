from logs import logDecorator as lD 
import json, pprint
from time import sleep
import numpy as np

from unityagents import UnityEnvironment

from lib.envs import envUnity

import gym
from gym import envs

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testUnity'

@lD.log( logBase + '.allTests' )
def allTests(logger):

	try:
		cfg  = json.load(open('../config/modules/tests.json'))['params']

		policy = lambda m: eval( cfg['UnityParams']['randomAction'] )

		with envUnity.Env(cfg['UnityParams']['binaryFile'], showEnv=True) as env:

			results = env.episode(policy, maxSteps = 1000)
			for i in  range(env.num_agents):
				print(f'For agent {i}:')
				s, a, r, ns, d = zip(*results[i])
				print(f'Rewards: {r}')
				print(f'Donnes: {d}')

	except Exception as e:
		logger.error('Unable to finish Unity tests: {}')


	return

