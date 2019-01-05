from logs import logDecorator as lD 
import json, pprint
from time import sleep

import gym
from gym import envs

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testOpenAI'

@lD.log( logBase + 'playOne' )
def playOne(logger, name):

	try:
		env = gym.make(name)
		env.reset()

		print('Example Action Space: ')
		print('--------------------- ')
		print(env.action_space)
		print('--------------------- ')


		for _ in range(50):
			env.render()
			observation, reward, done, info = env.step(env.action_space.sample())
			print(f'obssrvation: {observation}')
			print(f'reward     : {reward}')
			if done:
				break

		env.close()
		return True

	except Exception as e:
		return False

	return

@lD.log( logBase + 'renderOne' )
def renderOne(logger, name, sleepTime):

	try:
		env = gym.make(name)
		env.reset()
		env.render()
		sleep(sleepTime)
		env.close()
		return True

	except Exception as e:
		return False

	return

@lD.log( logBase + 'checkValidity' )
def checkValidity(logger, name):

	try:
		env = gym.make(name)
		env.reset()
		env.close()
		return True
	except Exception as e:
		return False

	return

@lD.log( logBase + '.allTests' )
def allTests(logger):

	try:
		cfg  = json.load(open('../config/modules/tests.json'))['params']
		if not cfg['TODO']['openAI']:
			print('OpenAI GYM environment is noto being tested')
			return

		cfg = cfg['openAI_params']

		if cfg['checkValidity']:
			print('Generating Registered Names:')
			allNames        = [str(k)[8:-1]  for k in envs.registry.all()]
			currName        = allNames[0]
			allNamesReduced = [currName] 

			currName = currName.split('-')[0]
			for name in allNames[1:]:
				if name.startswith(currName):
					continue
				allNamesReduced.append(name)
				currName = name.split('-')[0]

			print('Checking the validity of each name:')
			print('-----------------------------------')
			for name in allNamesReduced:
				if checkValidity(name):
					print('{:40s} : Valid'.format(name))
				else:
					print('{:40s} : Invalid'.format(name))

			print('-----------------------------------')

		if cfg['renderAll']:
			for name in allNamesReduced:
				renderOne(name, cfg['renderTime'])

		if cfg['renderOne'] is not None:
			renderOne(cfg['renderOne'], cfg['renderTime'])

		if cfg['playOne'] is not None:
			playOne(cfg['playOne'])

	except Exception as e:
		logger.error(f'Unable to complete the testing for OpenAAI gym: {e}')

	return
