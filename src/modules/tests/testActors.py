from logs import logDecorator as lD 
import json
import numpy as np

import torch
from torch.nn import functional as F

from lib.agents  import sequentialActor as sA, randomActor as rA

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testPolicy'

@lD.log( logBase + '.testArgMax' )
def testArgMax(logger):

    try:
        print('--------------------')

        sequence = sA.SequentialDiscreteActor(
            stateSize = 10, numActions = 3, 
            layers                  = [10, 5], 
            activations             = [F.tanh, F.tanh], 
            batchNormalization      = True)

        inp = np.random.rand(30, 10).astype( np.float32 ) * 10
        out = sequence( torch.as_tensor(inp) )


        print( out )
        print( torch.argmax(out, dim=1) )
    except Exception as e:
        logger.error(f'Unable to do the test for argmax: {e}')

    return

@lD.log( logBase + '.allTests' )
def allTests(logger):

    try:

        testArgMax()

    except Exception as e:
        logger.error(f'Unable to finish Memory Buffer tests: {e}')

    return

