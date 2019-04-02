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
from lib.envs    import envGym
from lib.utils   import ReplayBuffer as RB

from torch.nn import functional as F


config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.DQNagent.runAgent'

@lD.log(logBase + '.trainAgentUnity')
def trainAgentUnity(logger, configAgent):
    
    try:

        functionMaps = {
            'relu'      : F.relu,
            'relu6'     : F.relu6,
            'elu'       : F.elu,
            'celu'      : F.celu,
            'selu'      : F.selu,
            'prelu'     : F.prelu,
            'rrelu'     : F.rrelu,
            'glu'       : F.glu,
            
            'tanh'      : torch.tanh,
            'hardtanh'  : F.hardtanh

        }

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
        inpSize              = configAgent['inpSize']
        outSize              = configAgent['outSize']
        hiddenSizes          = configAgent['hiddenSizes']
        hiddenActivations    = configAgent['hiddenActivations']
        lr                   = configAgent['lr']

        hiddenActivations    = [functionMaps[m] for m in hiddenActivations]

        slidingScore = deque(maxlen=100)

        allResults = {
            "scores" : [],
            "slidingScores" : []
        }

        QNslow       = qN.qNetworkDiscrete( inpSize, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
        QNfast       = qN.qNetworkDiscrete( inpSize, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
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
def runAgentUnity(logger, configAgentUnity):
    

    allResults = trainAgentUnity(configAgentUnity)

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    toWrite = [now, json.dumps(configAgentUnity), json.dumps(allResults['scores']), json.dumps(allResults['slidingScores'])]
    
    with open(f'../results/agentDQN.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(toWrite)

    print(allResults)

    return

@lD.log(logBase + '.trainAgentGym')
def trainAgentGym(logger, configAgent):
    
    try:

        functionMaps = {
            'relu'      : F.relu,
            'relu6'     : F.relu6,
            'elu'       : F.elu,
            'celu'      : F.celu,
            'selu'      : F.selu,
            'prelu'     : F.prelu,
            'rrelu'     : F.rrelu,
            'glu'       : F.glu,
            
            'tanh'      : torch.tanh,
            'hardtanh'  : F.hardtanh

        }

        # Config parameters
        # --------------------------
        memorySize           = configAgent['memorySize']
        envName              = configAgent['envName']
        nIterations          = configAgent['nIterations']
        initMemoryIterations = configAgent['initMemoryIterations']
        eps0                 = configAgent['eps0']
        epsDecay             = configAgent['epsDecay']
        minEps               = configAgent['minEps']
        maxSteps             = configAgent['maxSteps']
        nSamples             = configAgent['nSamples']
        Tau                  = configAgent['Tau']
        inpSize              = configAgent['inpSize']
        outSize              = configAgent['outSize']
        hiddenSizes          = configAgent['hiddenSizes']
        hiddenActivations    = configAgent['hiddenActivations']
        lr                   = configAgent['lr']

        hiddenActivations    = [functionMaps[m] for m in hiddenActivations]

        slidingScore = deque(maxlen=100)

        allResults = {
            "scores" : [],
            "slidingScores" : []
        }

        QNslow       = qN.qNetworkDiscrete( inpSize, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
        QNfast       = qN.qNetworkDiscrete( inpSize, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
        
        with envGym.Env(envName, showEnv=True) as env:
            memoryBuffer = RB.SimpleReplayBuffer(memorySize)
            agent        = dqn.Agent_DQN(env, memoryBuffer, QNslow, QNfast, numActions=outSize, gamma=1, device='cuda:0')
            agent.eval()

            policy = lambda m: [agent.randomAction(m)]
            
            print('Generating some initial memory ...')
            for i in tqdm(range(initMemoryIterations)):
                score = agent.memoryUpdateEpisode(policy, maxSteps=maxSteps, minScoreToAdd = None)
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

@lD.log(logBase + '.runAgentGym')
def runAgentGym(logger, configAgentGym):
    

    allResults = trainAgentGym(configAgentGym)

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    toWrite = [now, json.dumps(configAgentGym), json.dumps(allResults['scores']), json.dumps(allResults['slidingScores'])]
    
    with open(f'../results/agentDQN.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(toWrite)

    print(allResults)

    return
