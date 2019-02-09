from logs import logDecorator as lD 
import json, pprint

import numpy               as np
import torch
import torch.nn.functional as F

from lib.agents import Agent_DQN as dqn
from lib.agents import qNetwork  as qN

from lib.envs    import envUnity
from lib.utils   import ReplayBuffer as RB


config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.testAgents.testAgents'


@lD.log(logBase + '.testAllAgents')
def testAllAgents(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    try:

        cfg          = json.load(open('../config/modules/testAgents.json'))['params']
        policy       = lambda m: eval( cfg['agentParams']['randomAction'] )
        memoryBuffer = RB.SimpleReplayBuffer(1000)
        QNslow       = qN.qNetworkDiscrete( 37, 4, [50, 30, 10], activations=[F.tanh, F.tanh, F.tanh] )
        QNfast       = qN.qNetworkDiscrete( 37, 4, [50, 30, 10], activations=[F.tanh, F.tanh, F.tanh] )


        with envUnity.Env(cfg['agentParams']['binaryFile'], showEnv=False) as env:

            agent = dqn.Agent_DQN(env, memoryBuffer, QNslow, QNfast, 4, 1)
            agent.eval()
            eps = 0.999
            policy = lambda m: [agent.epsGreedyAction(m, eps)]

            print('Starting to generate memories ...')
            print('----------------------------------------')
            for _ in range(3):
                print('[Generating Memories] ', end='', flush=True)
                score = agent.memoryUpdateEpisode(policy, maxSteps=1000)
                print( 'Memory Buffer lengths: {}\nScore: {}'.format( agent.memory.shape, score ) )

            print('Sampling from the memory:')
            memories = agent.memory.sample(20)
            s, a, r, ns, f = zip(*memories)
            s = np.array(s)
            
            print('Sampled some states of size {}'.format(s.shape))
            print('Finding the maxAction ....')
            s = torch.as_tensor(s.astype(np.float32))
            result1 = agent.randomAction(s)
            result2 = agent.maxAction(s)
            result3 = agent.epsGreedyAction(s, 0.5)
            
            print('Random Actioon stuff ......')
            results4 = env.episode(lambda m: [agent.randomAction(m)], 10)[0]
            # s, a, r, ns, f = zip(*results4)
            # print(s)

            print('Max Actioon stuff ......')
            results4 = env.episode(lambda m: [agent.maxAction(m)], 10)
            # print(len(results4))

            print('epsGreedy Actioon stuff ......')
            results4 = env.episode(lambda m: [agent.epsGreedyAction(m, 1)], 10)
            
            print('Load/Save a model')
            agent.save('../models', 'someName')
            agent.load('../models', 'someName')
            
            print('Doing a soft update')
            agent.step(nSamples=100)
            agent.softUpdate(0.2)
            print('Finished a soft update')

            # agent.step(nSamples = 10)
            
    except Exception as e:
        logger.error(f'Unable to test all agents: {e}')

    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''

    print('='*30)
    print('Main function of testAgents')
    print('='*30)

    testAllAgents()

    print('Getting out of testAgents')
    print('-'*30)

    return

