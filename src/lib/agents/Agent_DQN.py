import torch

class Agent_DQN:

    def __init__(self, env, memory, criticSlow, criticFast, numActions, randomAction, device='cpu'):

        if not torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'

        self.env          = env
        self.memory       = memory
        self.criticSlow   = criticSlow.to(device)
        self.criticFast   = criticFast.to(device)
        self.numAction    = numAction
        self.randomAction = randomAction

        self.eye          = torch.eye( numAction ).to(device)

        return

    def maxAction(self, state):

        states = torch.as_tensor(state).to(self.device).matmul( self.eye )
        qVals  = self.criticSlow( states, self.eye )

        qVals.argmax()

        return

    def memoryUpdateEpisode(self):

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