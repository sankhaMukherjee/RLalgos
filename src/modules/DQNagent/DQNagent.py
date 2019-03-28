from logs import logDecorator as lD 
import json, pprint, sys

from modules.DQNagent import runAgent

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.DQNagent.DQNagent'


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
    
    cConfig = json.load(open('../config/modules/DQNagent.json'))['params']
    configAgent = cConfig['configAgent'] 


    parameters = ['nIterations',
        'initMemoryIterations',
        'eps0',
        'epsDecay',
        'minEps',
        'maxSteps',
        'nSamples',
        'Tau',
        'lr']

    for p in parameters:
        try:
            configAgent[p] = resultsDict['dqnAgent'][p]
        except Exception as e:
            logger.warning(f' epsDecay not set from the input: {e}')


    print('\n\n+--------------------------------------')
    print('| Current Configuration')
    print('+--------------------------------------')
    for c, v in configAgent.items():
        print(f'| [{c:30s}] --> {v}')
    print('+--------------------------------------')

    runAgent.runAgent(configAgent)

    print('Getting out of DQNagent')
    print('-'*30)

    return

