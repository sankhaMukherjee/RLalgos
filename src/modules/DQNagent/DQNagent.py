from logs import logDecorator as lD 
import json, pprint, sys

from collections import deque

from tqdm import tqdm

import numpy               as np
import torch
import torch.nn.functional as F

from lib.agents import Agent_DQN as dqn
from lib.agents import qNetwork  as qN

from lib.envs    import envUnity
from lib.utils   import ReplayBuffer as RB

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.DQNagent.DQNagent'


@lD.log(logBase + '.trainAgent')
def trainAgent(logger):
    
    try:

        # Config parameters
        # --------------------------
        memorySize           = 100000
        binary               = "/home/sankha/data/UnityBinaries/UdacityBinaries/Banana_Linux/Banana.x86_64"
        nIterations          = 2000
        initMemoryIterations = 100
        eps0                 = 1 # Probability that max action will happen
        epsDecay             = 0.995
        minEps               = 0.01
        maxSteps             = 1000
        nSamples             = 1000
        Tau                  = 0.1

        slidingScore = deque(maxlen=100)

        with envUnity.Env(binary, showEnv=False) as env:
            memoryBuffer = RB.SimpleReplayBuffer(memorySize)
            QNslow       = qN.qNetworkDiscrete( 37, 4, [64, 64], activations=[F.tanh, F.tanh] )
            QNfast       = qN.qNetworkDiscrete( 37, 4, [64, 64], activations=[F.tanh, F.tanh] )
            agent        = dqn.Agent_DQN(env, memoryBuffer, QNslow, QNfast, numActions=4, gamma=1, device='cuda:0')
            agent.eval()

            policy = lambda m: [agent.randomAction(m)]
            
            print('Generating some initial memory ...')
            for i in tqdm(range(initMemoryIterations)):
                score = agent.memoryUpdateEpisode(policy, maxSteps=maxSteps, minScoreToAdd = 1)
                tqdm.write(f'score = {score}')

            eps = eps0
            print('Optimizing model ...')
            for i in tqdm(range(nIterations)):
                
                eps = max(minEps, epsDecay*eps) # decrease epsilon

                policy = lambda m: [agent.epsGreedyAction(m, 1-eps)]
                agent.memoryUpdateEpisode(policy, maxSteps=maxSteps)

                agent.step(nSamples=nSamples)
                agent.softUpdate(Tau)
                
                # Calculate Score
                results = env.episode(lambda m: [agent.maxAction(m)], 10)[0]
                s, a, r, ns, f = zip(*results)
                score = sum(r)
                slidingScore.append(score)
                tqdm.write('score = {}, max = {}, sliding score = {}'.format(score, max(r), np.mean( slidingScore ) ))

    except Exception as e:
        logger.error(f'Unable to train the agent: {e}')

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
    print('Main function of DQNagent')
    print('='*30)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)

    trainAgent()

    print('Getting out of DQNagent')
    print('-'*30)

    return

