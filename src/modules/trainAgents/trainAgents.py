from logs import logDecorator as lD 
import json, pprint

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.trainAgents.trainAgents'

from lib.agents import trainAgents as tA

@lD.log(logBase + '.doSomething')
def doSomething(logger, resultsDict):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    configAgentGym = {

        "envName": "CartPole-v1",

        "memorySize": 100005,
        "initMemoryIterations": 1000,
        "updateMemory": [],

        "nIterations": 1000,

        "eps0": 1,
        "epsDecay": 0.99,
        "minEps": 0.01,

        "maxSteps": 10000,
        "nSamples": 500,
        "Tau": 0.001,

        "N": 4,
        "inpSize": 4,
        "outSize": 2,
        "hiddenSizes": [256, 128, 128],
        "hiddenActivations": ["tanh", "tanh", "tanh"],
        "lr": 0.001,

        "loadFolder": None,
        "saveFolder": None
    }

    
    configAgentGym['saveFolder'] = f'/home/sankha/Documents/mnt/hdd01/models/RLalgos/{configAgentGym["envName"]}'
    configAgentGym['loadFolder'] = f'/home/sankha/Documents/mnt/hdd01/models/RLalgos/CartPole-v1/2019-06-02--11-27-13_00024_111'

    parameters = ['nIterations',
                  'initMemoryIterations',
                  'eps0',
                  'epsDecay',
                  'minEps',
                  'maxSteps',
                  'nSamples',
                  'Tau',
                  "inpSize",
                  "outSize",
                  "hiddenSizes",
                  "hiddenActivations",
                  'lr']

    for p in parameters:
        try:
            configAgent[p] = resultsDict['dqnAgent'][p]
        except Exception as e:
            logger.warning(f' epsDecay not set from the input: {e}')

    results = tA.trainAgentGymEpsGreedy(configAgentGym)
    for r in results:
        print(f'-----------[{r.center(40)}]--------------')
        print( results[r] )

    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for trainAgents
    
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
    print('Main function of trainAgents')
    print('='*30)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)

    doSomething(resultsDict)

    print('Getting out of trainAgents')
    print('-'*30)

    return

