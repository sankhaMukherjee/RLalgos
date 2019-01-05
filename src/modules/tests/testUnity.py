from logs import logDecorator as lD 
import json, pprint
from time import sleep
import numpy as np

from unityagents import UnityEnvironment

import gym
from gym import envs

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testUnity'

@lD.log( logBase + '.allTests' )
def allTests(logger):

	try:
		cfg  = json.load(open('../config/modules/tests.json'))['params']
		if not cfg['TODO']['Unity']:
			print('The Unity environment is noto being tested')
			return

		print('Doing Unity Tests')
		cfg = cfg['UnityParams']

		env = UnityEnvironment(file_name = cfg['binaryFile'])
		brain_name = env.brain_names[0]
		env_info = env.reset(train_mode=False)[brain_name]

		num_agents = len(env_info.agents)
		print('Number of agents:', num_agents)

		for i in range(100):
			env_info  = env.step( eval(cfg['randomAction']) )[brain_name]
			o, r, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]

			print(o, r, done)
			if done:
				break


	except Exception as e:
		logger.error('Unable to finish Unity tests')


	return

