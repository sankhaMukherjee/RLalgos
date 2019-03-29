from logs import logDecorator as lD 
import json, pprint
import numpy as np

from tqdm import tqdm

from lib.envs    import envUnity
from lib.utils   import ReplayBuffer as RB

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.playAgent.playAgent'

import sys,tty,termios
class _Getch:       
    def __call__(self):
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch


def get():
    inkey = _Getch()
    while(1):
            k=inkey()
            if k!='':break
    
    return k


def policy(m):
    mapping  = {
        'w': [0], # a
        's': [1], # s
        'a': [2], # d
        'd': [3], # w

    }

    while True:
        k = get()
        return mapping.get(k, [np.random.randint(0, 4)])

    
    return [np.random.randint(0, 4)]

@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    cConfig = json.load(open('../config/modules/playAgent.json'))['params']
    print(cConfig)

    memoryBuffer = RB.SimpleReplayBuffer(cConfig['memorySize'])
    
    with envUnity.Env(cConfig['binary'], showEnv=True, trainMode=False) as env:
            
        results = env.episode( policy , cConfig['maxSteps'])[0]
        s, a, r, ns, f = zip(*results)
        print(r)


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
    print('Main function of playAgent')
    print('='*30)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)

    doSomething()

    print('Getting out of playAgent')
    print('-'*30)

    return

