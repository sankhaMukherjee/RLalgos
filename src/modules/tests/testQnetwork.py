from logs import logDecorator as lD 
import json
import numpy as np

import torch
from torch.nn import functional as F

from lib.agents import qNetwork  as qN
import torch

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.testPolicy'

@lD.log( logBase + '.testArgMax' )
def testQnetwork(logger):

    try:
        print('--------------------')

        qn = qN.qNetworkDiscrete( 2, 1, [64, 64], activations=[torch.tanh, torch.tanh], lr=0.001)

        X    = np.random.rand(30, 2).astype( np.float32 ) 
        y    = (2*X[:,0] + 3*X[:,1]).reshape(-1, 1)
        yT   = torch.as_tensor( y )

        for i in range(1000):
            qn.step( qn( torch.as_tensor(X) ), yT )

            if i % 100 == 0:
                yHat = qn( torch.as_tensor(X) ).cpu().detach().numpy()
                print(np.mean((yHat - y)**2))

        yHat = qn( torch.as_tensor(X) ).cpu().detach().numpy()
        print(f'Final error: {np.mean((yHat - y)**2)}')
        
        
    except Exception as e:
        logger.error(f'Unable to do the test for argmax: {e}')

    return

@lD.log( logBase + '.allTests' )
def allTests(logger):

    try:

        testQnetwork()

    except Exception as e:
        logger.error(f'Unable to finish Memory Buffer tests: {e}')

    return

