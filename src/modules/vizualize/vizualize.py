from logs import logDecorator as lD 
import json, pprint, os

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.vizualize.vizualize'

from lib.agents import Agent_DQN as dqn
from lib.agents import qNetwork as qN

from lib.envs import envUnity
from lib.envs import envGym
from lib.utils import ReplayBuffer as RB

import torch
from torch.nn import functional as F


@lD.log(logBase + '.doSomething')
def doSomething(logger, folder):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    configAgent = json.load( open(os.path.join(folder, 'configAgent.json')) )

    memorySize           = configAgent['memorySize']
    envName              = configAgent['envName']
    maxSteps             = 10000
    inpSize              = configAgent['inpSize']
    outSize              = configAgent['outSize']
    hiddenSizes          = configAgent['hiddenSizes']
    hiddenActivations    = configAgent['hiddenActivations']
    lr                   = configAgent['lr']
    N                    = configAgent['N']
    loadFolder           = folder

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
        'hardtanh': F.hardtanh }

    hiddenActivations = [functionMaps[m] for m in hiddenActivations]

    
    QNslow = qN.qNetworkDiscrete(
        inpSize*N, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
    QNfast = qN.qNetworkDiscrete(
        inpSize*N, outSize, hiddenSizes, activations=hiddenActivations, lr=lr)
    memoryBuffer = RB.SimpleReplayBuffer(memorySize)

    with envGym.Env1D(envName, N=N, showEnv=True) as env:
        agent = dqn.Agent_DQN(
            env, memoryBuffer, QNslow, QNfast, numActions=outSize, gamma=1, device='cuda:0')
        agent.load(loadFolder, 'agent_0')

        def policy(m): 
            return [agent.maxAction(m)]

        for i in range(10):
            env.reset()
            allResults = env.episode(policy, maxSteps=maxSteps)
            s, a, r, ns, f = zip(*allResults[0])
            
            actions = ''.join([str(m) for m in a])
            print(sum(r), actions)
            
    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for vizualize
    
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
    print('Main function of vizualize')
    print('='*30)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)

    folder = f'/home/sankha/Documents/mnt/hdd01/models/RLalgos/CartPole-v1/2019-06-02--17-01-22_00487_500/'
    doSomething(folder)

    print('Getting out of vizualize')
    print('-'*30)

    return

