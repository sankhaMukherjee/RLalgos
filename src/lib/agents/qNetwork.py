
import numpy as np
import sys
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim

class qNetworkDiscrete(nn.Module):

    def __init__(self, stateSize, actionSize, layers=[10, 5], activations=[F.tanh, F.tanh], batchNormalization = False, lr=0.01 ):
        '''This is a Q network with discrete actions
        
        This takes a state and returns a Q function for each action. Hence, the
        input is a state and the output is a set of Q values, one for each action
        in the action space. The action is assumed to be discrete. i.e. a ``1``
        when the particular action is to be desired. The input state is assumed to
        be 1D in nature. A different network will have to be chosen if 2D and 3D 
        inputs are to be desired.
        
        Parameters
        ----------
        stateSize : {int}
            Size of the state. Since this is a 1D network, this represents the number
            of values will be used to represent the current state. 
        actionSize : {int}
            The number of discrete actions that will be used.
        layers : {list of int}, optional
            The number of nodes associated with each layer (the default is ``[10, 5]``
            , which will create two hidden layers with and and 5 nodes each)
        activations : {list of activations}, optional
            The activation functions to be used for each layer (the default is 
            ``[F.tanh, F.tanh]``, which will generate tanh activations for 
            each of the hidden layers)
        batchNormalization : {bool}, optional
            Whether batchnormalization is to be used (the default is ``False``,
            for which batch normalization will be neglected)
        '''


        try:
            super(qNetworkDiscrete, self).__init__()
            self.stateSize           = stateSize
            self.actionSize          = actionSize
            self.layers              = layers
            self.activations         = activations
            self.batchNormalization  = batchNormalization

            # Generate the fullly connected layer functions
            self.fcLayers = nn.ModuleList([])
            self.bns = nn.ModuleList([])

            oldN = stateSize
            if self.batchNormalization:
                for i, layer in enumerate(layers):
                    self.fcLayers.append( nn.Linear(oldN, layer) )
                    self.bns.append( nn.BatchNorm1d( num_features = layer, track_running_stats=True ) )
                    oldN = layer
            else:
                for i, layer in enumerate(layers):
                    self.fcLayers.append( nn.Linear(oldN, layer) )
                    oldN = layer

            # ------------------------------------------------------
            # The final layer will only need to supply a quality
            # function. This is a single value for each action 
            # provided. Ideally, you would want to provide a 
            # OHE action sequence for most purposes ...
            # ------------------------------------------------------
            self.fcFinal = nn.Linear( oldN, actionSize )

            # we shall put this is eval mode and only use 
            # the trian mode when we need to train the 
            # mode
            self.optimizer = optim.Adam(
                self.parameters(), lr=lr)
        
        except Exception as e:
            raise type(e)( 
                'lib.agents.qNetwork.qNetworkDiscrete.__init__ - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return

    def forward(self, x, sigma=0):
        '''forward function that is called during the forward pass
        
        This is the forward function that will be called during a 
        forward pass. It takes thee states and gives the Q value 
        correspondidng to each of the applied actions that are 
        associated with that state. 
        
        Parameters
        ----------
        x : Tensor
            This is a 2D tensor. 
        
        Returns
        -------
        tensor
            This represents the Q value of the function
        '''

        try:
            if self.batchNormalization:
                for i, (bn, fc, a) in enumerate(zip(self.bns, self.fcLayers, self.activations)):
                    if self.training:
                        bn.train()
                    else:
                        bn.eval()
                    x = a(bn(fc(x)))
                    # https://discuss.pytorch.org/t/random-number-on-gpu/9649
                    if x.is_cuda:
                        normal = torch.cuda.FloatTensor(x.shape).normal_()
                    else:
                        normal = torch.FloatTensor(x.shape).normal_()
                    x = x + normal*sigma

                x = self.fcFinal( x )

            else:
                for i, (fc, a) in enumerate(zip(self.fcLayers, self.activations)):
                    x = a(fc(x))
                    # https://discuss.pytorch.org/t/random-number-on-gpu/9649
                    if x.is_cuda:
                        normal = torch.cuda.FloatTensor(x.shape).normal_()
                    else:
                        normal = torch.FloatTensor(x.shape).normal_()
                    x = x + normal*sigma

                x = self.fcFinal( x )

        except Exception as e:
            raise type(e)( 
                'lib.agents.qNetwork.qNetworkDiscrete.forward - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return x

    def step(self, v1, v2):
        '''Uses the optimizer to update the weights
        
        This calculates the MSE loss given two inputs,
        one of which must be calculated with this current
        ``nn.Module``, and the other one that is expected.
        
        Note that this allows arbitrary functions to be used
        for calculating the loss.
        
        Parameters
        ----------
        v1 : {Tensor}
            Tensor for calculating the loss function
        v2 : {Tensor}
            Tensor for calculating the loss function
        
        Raises
        ------
        type
            [description]
        '''

        try:
            loss = F.mse_loss(v1, v2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            raise type(e)( 
                'lib.agents.qNetwork.qNetworkDiscrete.forward - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return
        
