from logs import logDecorator as lD 
import json, csv

from collections import deque

from tqdm import tqdm

import numpy               as np
import torch
from datetime import datetime as dt

from lib.agents import Agent_DQN as dqn
from lib.agents import qNetwork  as qN

from lib.envs    import envUnity
from lib.utils   import ReplayBuffer as RB


config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.DQNagent.runAgent'

@lD.log(logBase + '.trainAgent')
def trainAgent(logger, configAgent):
    
    try:

        # Config parameters
        # --------------------------
        memorySize           = configAgent['memorySize']
        binary               = configAgent['binary']
        nIterations          = configAgent['nIterations']
        initMemoryIterations = configAgent['initMemoryIterations']
        eps0                 = configAgent['eps0']
        epsDecay             = configAgent['epsDecay']
        minEps               = configAgent['minEps']
        maxSteps             = configAgent['maxSteps']
        nSamples             = configAgent['nSamples']
        Tau                  = configAgent['Tau']
        lr                   = configAgent['lr']

        slidingScore = deque(maxlen=100)

        allResults = {
            "scores" : [],
            "slidingScores" : []
        }

        QNslow       = qN.qNetworkDiscrete( 37, 4, [64, 64], activations=[torch.tanh, torch.tanh], lr=lr)
        QNfast       = qN.qNetworkDiscrete( 37, 4, [64, 64], activations=[torch.tanh, torch.tanh], lr=lr)
        with envUnity.Env(binary, showEnv=False) as env:
            memoryBuffer = RB.SimpleReplayBuffer(memorySize)
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

                policy = lambda m: [agent.epsGreedyAction(m, eps)]
                agent.memoryUpdateEpisode(policy, maxSteps=maxSteps)

                agent.step(nSamples=nSamples)
                agent.softUpdate(Tau)
                
                # Calculate Score
                results = env.episode(lambda m: [agent.maxAction(m)], maxSteps)[0]
                s, a, r, ns, f = zip(*results)
                score = sum(r)
                slidingScore.append(score)
                tqdm.write('score = {}, max = {}, sliding score = {}, eps = {}'.format(score, max(r), np.mean( slidingScore ), eps ))

                allResults['scores'].append( score )
                allResults['slidingScores'].append( np.mean(slidingScore) )

            # env.env.close()

        return allResults

    except Exception as e:
        logger.error(f'Unable to train the agent: {e}')

    return

@lD.log(logBase + '.runAgent')
def runAgent(logger, configAgent):
    

    allResults = trainAgent(configAgent)

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    toWrite = [now, json.dumps(configAgent), json.dumps(allResults['scores']), json.dumps(allResults['slidingScores'])]
    
    with open('../results/agentDQN.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(toWrite)

    print(allResults)

    return