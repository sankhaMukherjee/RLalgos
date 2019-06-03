import json
import csv, sys, os

from collections import deque

from tqdm import tqdm

import numpy as np
import torch
from datetime import datetime as dt

from lib.agents import Agent_DQN as dqn
from lib.agents import Agent_DoubleDQN as ddqn
from lib.agents import qNetwork as qN

from lib.envs import envUnity
from lib.envs import envGym
from lib.utils import ReplayBuffer as RB

from torch.nn import functional as F

from datetime import datetime as dt


def trainAgentGymEpsGreedy(configAgent):

    try:

        now = dt.now().strftime(r'%Y-%m-%d--%H-%M-%S')

        functionMaps = {
            'relu': F.relu,
            'relu6': F.relu6,
            'elu': F.elu,
            'celu': F.celu,
            'selu': F.selu,
            'prelu': F.prelu,
            'rrelu': F.rrelu,
            'glu': F.glu,

            'tanh': torch.tanh,
            'hardtanh': F.hardtanh

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
        N                    = configAgent['N']
        sigma                = configAgent['sigma']
        loadFolder           = configAgent['loadFolder']
        saveFolder           = configAgent['saveFolder']

        hiddenActivations = [functionMaps[m] for m in hiddenActivations]

        slidingScore = deque(maxlen=100)
        prevBest = -np.inf

        allResults = {
            "scores": [],
            "slidingScores": [],
            "saveLocations": [],
        }

        QNslow = qN.qNetworkDiscrete(
            inpSize*N, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
        QNfast = qN.qNetworkDiscrete(
            inpSize*N, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
        memoryBuffer = RB.SimpleReplayBuffer(memorySize)

        with envGym.Env1D(envName, N=N, showEnv=False) as env:

            if configAgent['agentType'] == 'DQN':
                agent = dqn.Agent_DQN(
                    env, memoryBuffer, QNslow, QNfast, numActions=outSize, gamma=1, device='cuda:0')
            if configAgent['agentType'] == 'DoubleDQN':
                agent = ddqn.Agent_DoubleDQN(
                    env, memoryBuffer, QNslow, QNfast, numActions=outSize, gamma=1, device='cuda:0')

            if loadFolder:
                agent.load( loadFolder, 'agent_0' )
            agent.eval()


            if not loadFolder:
                def policy(m): return [agent.sigmaMaxAction(m, 1)]
            else:
                def policy(m): return [agent.sigmaMaxAction(m, 0.01)]
                agent.memory.clear()

            agent.fastUpdate()

            print('Generating some initial memory ...')
            for i in tqdm(range(initMemoryIterations)):
                score = agent.memoryUpdateEpisode(
                    policy, maxSteps=maxSteps, minScoreToAdd=None)
                tqdm.write(f'score = {score}')

            eps = eps0
            print('Optimizing model ...')
            for i in tqdm(range(nIterations)):

                eps = max(minEps, epsDecay*eps)  # decrease epsilon

                # We are changing the policy to adding noise
                def policy(m): return [agent.sigmaMaxAction(m, eps*3)]
                agent.memoryUpdateEpisode(policy, maxSteps=maxSteps)

                agent.step(nSamples=nSamples, sigma=sigma)
                agent.softUpdate(Tau)

                # Calculate Score
                results = env.episode(
                    lambda m: [agent.maxAction(m)], maxSteps)[0]
                s, a, r, ns, f = zip(*results)
                score = sum(r)
                slidingScore.append(score)

                if (score > prevBest):
                    tqdm.write('score = {}, max = {}, sliding score = {}, eps = {}'.format(
                        score, max(r), np.mean(slidingScore), eps))

                if saveFolder and (score > prevBest):
                    prevBest = score
                    folder = os.path.join( saveFolder, f'{now}_{i:05d}_{int(score)}' )
                    os.makedirs(folder)
                    agent.save(folder, 'agent_0')
                    allResults['saveLocations'].append((score, folder))

                    json.dump(configAgent, open(os.path.join(folder, 'configAgent.json'), 'w'))

                allResults['scores'].append(score)
                allResults['slidingScores'].append(np.mean(slidingScore))

        return allResults

    except Exception as e:
        raise type(e)(
            'lib.agents.Agent_DQN.Agent_DQN.save - ERROR - ' + str(e)
        ).with_traceback(sys.exc_info()[2])

    return
