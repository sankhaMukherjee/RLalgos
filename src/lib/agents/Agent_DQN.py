import torch
import numpy as np 

class Agent_DQN:

    def __init__(self, env, memory, qNetworkSlow, qNetworkFast, numActions, device='cpu'):

        if not torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'

        self.env            = env
        self.memory         = memory
        self.qNetworkSlow   = qNetworkSlow.to(device)
        self.qNetworkFast   = qNetworkFast.to(device)
        self.numActions     = numActions

        return

    def randomAction(self, state):
        '''returns a set of random actions for the given states
        
        given the size of the number of actions, this function is going
        to return a set of actions that has the same number of actions
        as the number of inputs in the shape. For example, if 
        ``state.shape == (10, ?)`` then the result will be a vector of
        size 10. This is in accordance with the redduction in the
        dimensionality of the maxAction space. 
        
        Parameters
        ----------
        state : {nd_array or tensor}
            numpy array or tensor containing the state. The columns
            represent the different parts of the state.
        
        Returns
        -------
        uarray
            The return value is set of random actions
        '''
        r, c = state.shape
        result = np.random.randint(0, self.numActions, size=r)
        return result

    def maxAction(self, state):
        '''returns the action that maximizes the Q function
        
        Given an set of statees, this function is going to return a set
        of actions which will maximize the value of the Q network for each
        of the supplied states.
        
        Parameters
        ----------
        state : {nd_array or tensor}
            numpy array or tensor containing the state. The columns
            represent the different parts of the state.
        
        Returns
        -------
        uarray
            The return values of actions that maximize the states
        '''
        state  = torch.as_tensor(state).to(self.device)
        qVals  = self.qNetworkSlow( state )
        result = torch.argmax(qVals, dim=1)
        return result

    def memoryUpdateEpisode(self, policy, maxSteps=1000):
        '''update the memory
        
        Given a particular policy, this memory is going to take
        the policy and generate a series of memories and update
        thememory buffer. Generating memories is easier to do 
        using this function than an external function ...
        
        Parameters
        ----------
        policy : {function}
            This is a function that takes a state and returns an action. This
            defines how the agent will explore the environment by changing the
            exploration/exploitation scale.
        maxSteps : {number}, optional
            The maximum number of steps that one shoule have within an episode. 
            (the default is 1000)
        '''
        allResults = self.env.episode(policy, maxSteps = maxSteps)
        self.memory.appendAllAgentResults( allResults )
        return

    def step(self):

        return

    def softUpdate(self, Tau=0.1):

        return

    def updateBuffer(self, data):

        return

    def save(self, folder, name):

        return

    def load(self, folder, name):

        return