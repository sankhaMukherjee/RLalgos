import torch, os
import numpy as np 
import sys

class Agent_DoubleDQN:

    def __init__(self, env, memory, qNetworkSlow, qNetworkFast, numActions, gamma, device='cpu'):
        '''A class allowing the training of the DQN

        This class is intended to be used by functions within the ``lib.agents.trainAgents``
        module.
        
        This is supposed to be an imporvement over DQN. Details of the idea behind Double DQN is
        present in the original paper: 

        Deep Reinforcement Learning with Double Q-learning
        https://arxiv.org/pdf/1509.06461.pdf

        See the ``step()`` function for the details of its implementation.

        Parameters
        ----------
        env : instance of an Env class
            The environment that will be used for generating the result of a particulat action
            in the current state
        memory : instance of the Memory class
            The environment that will allow one to store and retrieve previously held states that
            can be used to train upon.
        qNetworkSlow : neural network instance
            This is a neural network instance that can be used for converting a state into a
            set of Q-values. This is the slower version, used for making a prediction, and is 
            never trained. Its parameters are slowly updated over time to slowly allow it to 
            converge to the right value
        qNetworkFast : neural network instance
            This is the instance of the faster network that can be used for training Q-learning
            algorithm. This is the main network that implements the Bellman equation.
        numActions : int
            The number of discrete actions that the current environment can accept.
        gamma : float
            The discount factor. currently not used
        device : str, optional
            the device where you want to run your algorithm, by default 'cpu'. If you want to run
            the optimization of a particular GPU, you may specify that. For example with 'cuda:0'
        
        Raises
        ------
        type
            [description]
        '''
        
        try:

            if not torch.cuda.is_available():
                self.device = device
            else:
                self.device = 'cpu'

            self.env            = env
            self.memory         = memory
            self.qNetworkSlow   = qNetworkSlow.to(self.device)
            self.qNetworkFast   = qNetworkFast.to(self.device)
            self.gamma          = torch.as_tensor(gamma).float().to(device)
            self.numActions     = numActions
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.__init__ - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

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

        try:
            r, c = state.shape
            result = np.random.randint(0, self.numActions, size=r).astype(np.float32)
            result = torch.as_tensor(result).to(self.device)
            return result
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.randomAction - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

    def sigmaMaxAction(self, state, sigma=0):
        '''returns the action that maximizes the noisy Q function 
        
        Given an set of statees, this function is going to return a set
        of actions which will maximize the value of the Q network for each
        of the supplied states, after adding Gaussian noise to the layers.
        This is alternative to using an $\\epsilon$-greedy policy, and has 
        shown to provide better results under most circumstances.
        
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

        try:

            state = torch.as_tensor(state).float().to(self.device)
            qVals = self.qNetworkSlow(state, sigma)
            result = torch.argmax(qVals, dim=1)
            result = result.to(dtype=torch.float32, device=self.device)
            return result
        except Exception as e:
            raise type(e)(
                'lib.agents.Agent_DQN.Agent_DQN.maxAction - ERROR - ' + str(e)
            ).with_traceback(sys.exc_info()[2])

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

        try:

            state  = torch.as_tensor(state).float().to(self.device)
            qVals  = self.qNetworkSlow( state )
            result = torch.argmax(qVals, dim=1)
            result = result.to(dtype=torch.float32, device=self.device)
            return result
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.maxAction - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

    def epsGreedyAction(self, state, eps=0.999):
        '''epsilon greedy action
        
        This is the epsilon greedy action. In general, this is going to
        select the maximum action ``eps`` percentage of the times, while
        selecting the random action the rest of the time. It is assumed
        that this will receive a value of epsilon between 0 and 1.
        
        Parameters
        ----------
        state : {ndarray}
            [description]
        eps : float, optional
            Determines the fraction of times the max action will be selected
            in comparison to a random action. (the default is 0.999)
        
        Returns
        -------
        tensor
            The 1d tensor that has an action for each state provided. 
        '''

        try:
            ma = self.maxAction(state)
            ra = self.randomAction(state)
            p  = np.random.choice([1, 0], size=len(ma), p=[1-eps, eps]).astype(np.float32)
            p  = torch.as_tensor( p ).to(self.device)

            result = ma * p + ra * (1-p)

            return result
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.epsGreedyAction - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

    def memoryUpdateEpisode(self, policy, maxSteps=1000, minScoreToAdd=None):
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

        try:
            allResults = self.env.episode(policy, maxSteps = maxSteps)
            s, a, r, ns, f = zip(*allResults[0])
            score = np.sum(r)
            if (minScoreToAdd is None):
                self.memory.appendAllAgentResults( allResults )

            if (minScoreToAdd is not None) and (score >= minScoreToAdd):
                self.memory.appendAllAgentResults( allResults )
            return score
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.memoryUpdateEpisode - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

    def step(self, nSamples = 100, sigma=0):
        '''optimize the fast Q network via the bellman equations

        This function is going to obtain a number of samples form the 
        replay memory, and train the fast network over this dataset. 
        The idea behind this is that the Q network for the next step 
        will not automatically pick the next best value. It will possibly
        pick the best value that the Fast network thinks it will pick, and
        so the original DQN will overestimate the possible Q value. This 
        should reduce that estimation.
        
        This will optimize based upon the idea that
        
        > Given:
        >     Qf = fast network
        >     Qs = slow network
        >     s  = current state
        >     a  = action that maximizes the current state
        >     r  = reward
        >     s' = next state
        > 
        > a'  = argmax Qf(s')
        > Qf(s, a) = r + Qs(s', a')
        
        Parameters
        ----------
        nSamples : int, optional
            The number of samples too retrieve from the replay memory for 
            training, by default 100
        sigma : float, optional
            The amount by which the fast network should be jittered so that the
            network introduces some Gaussian noise in the learning process, by 
            default 0, which does not introduce noise in the learning algorithm.
        
        Raises
        ------
        type
            [description]
        '''

        try:


            data = self.memory.sample( nSamples )
            states, actions, rewards, nextStates, dones = zip(*data)
            
            states      = torch.as_tensor(states).float().to(self.device)
            actions     = torch.as_tensor(actions).float().to(self.device)
            rewards     = torch.as_tensor(rewards).float().to(self.device)
            nextStates  = torch.as_tensor(nextStates).float().to(self.device)
            dones       = torch.as_tensor(dones).float().to(self.device)
            
            # We cannot just assume that the best action will be the one
            # taken from the max of the next network. It is infact the 
            # one that is chosen by the fast network for the next step.
            newQVals = self.qNetworkFast(nextStates)
            newActions = torch.argmax(newQVals, dim=1)
            # print(newActions)
            # print(self.qNetworkSlow(nextStates)[:10])
            # print(self.qNetworkSlow(nextStates).shape)
            # print(self.qNetworkSlow(nextStates)[:, newActions][:, 0].shape)

            self.qNetworkFast.train()
            # self.qNetworkSlow.train()

            # Note that `max` also returns the positions
            qVal    = self.qNetworkFast( states, sigma ).max(dim=1)[0]
            qValHat = rewards + self.qNetworkSlow( nextStates )[ :, newActions ][:, 0] * (1-dones)
            
            self.qNetworkFast.step(qValHat, qVal)
            
            self.qNetworkFast.eval()
            # self.qNetworkSlow.eval()
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.step - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])
        
        return

    def checkTrainingMode(self):
        '''prints whether the networks are in training or eval mode
        
        This function allows us to determine whether the function is 
        in training or evaluation mode. This is important for several
        things -  specifically to make sure that the networks are not
        going to be randomly evaluated, as well as making sure the 
        things like batch normalization and dropout are properly evaluated.
        '''
        try:
            print('qNetworkSlow is in trai mode:', self.qNetworkSlow.training)
            print('qNetworkFast is in trai mode:', self.qNetworkFast.training)
            return
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.checkTrainingMode - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

    def eval(self):
        '''put both the networks in eval mode
        
        This will allow us to make sure that the networks
        do not randomly get evaluated for some reason.
        '''
        try:
            self.qNetworkFast.eval()
            self.qNetworkSlow.eval()
            return
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.eval - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

    def softUpdate(self, tau=0.1):
        '''update the slow network slightly
        
        This is going to update the slow network slightly. The amount
        is dictated by ``tau``. This should be a number between 0 and 1.
        It will update the ``tau`` fraction of the slow network weights
        with the new weights. This is done for providing stability to the
        network. 
        
        Parameters
        ----------
        tau : {number}, optional
            This parameter determines how much of the fast Networks weights
            will be updated to the ne parameters weights (the default is 0.1)
        '''

        for v1, v2 in zip(self.qNetworkFast.parameters(), self.qNetworkSlow.parameters()):
            v2.data.copy_( tau*v1 + (1-tau)*v2 )

        return

    def fastUpdate(self, tau=1):
        '''update the fast network slightly
        
        This is going to update the fast network slightly. The amount
        is dictated by ``tau``. This should be a number between 0 and 1.
        It will update the ``tau`` fraction of the slow network weights
        with the new weights. This is done for providing stability to the
        network. 
        
        Parameters
        ----------
        tau : {number}, optional
            This parameter determines how much of the slow Networks weights
            will be updated to the fast parameters weights (the default is 1)
        '''

        for v1, v2 in zip(self.qNetworkFast.parameters(), self.qNetworkSlow.parameters()):
            v1.data.copy_(tau*v2 + (1-tau)*v1)

        return

    def save(self, folder, name):
        '''save the model
        
        This function allows one to save the model, in a folder that is 
        specified, with the fast and the slow qNetworks, as well as the
        memory buffer. Sometimes there may be more than a single agent,
        and under those circumstances, the name will come in handy. If the
        supplied folder does not exist, it will be generated. 
        
        Parameters
        ----------
        folder : {str}
            folder into which the model should be saved.
        name : {str}
            A name to associate the current model with. It is
            absolutelty possible to save a number of models within
            the same folder.
        '''

        try:
            if not os.path.exists(folder):
                os.makedirs(folder)

            torch.save(
                self.qNetworkFast.state_dict(), 
                os.path.join(folder, f'{name}.qNetworkFast'))

            torch.save(
                self.qNetworkSlow.state_dict(), 
                os.path.join(folder, f'{name}.qNetworkSlow'))

            self.memory.save(folder, name)
            
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.save - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return

    def load(self, folder, name, map_location = None):
        '''load the model
        
        An agent saved with the save command can be safely loaded with this command.
        This will load both the qNetworks, as well as the memory buffer. There is a
        possibility that one may not want to load the model into the same device. In
        that case, the user should insert the device that the user wants to load the
        model into. 
        
        Parameters
        ----------
        folder : {str}
            folder into which the model should be saved.
        name : {str}
            A name to associate the model to load. It is absolutelty possible to save 
            a number of models within the same folder, and hence the name can retrieve
            that model that is important.
        map_location : {str}, optional
            The device in which to load the file. This is a string like 'cpu', 'cuad:0'
            etc. (the default is None, which results in the model being loaded to the 
            originam device)
        '''

        try:
            if map_location is None:
                self.qNetworkSlow.load_state_dict(
                    torch.load(os.path.join(folder, f'{name}.qNetworkSlow')))
                self.qNetworkFast.load_state_dict(
                    torch.load(os.path.join(folder, f'{name}.qNetworkFast')))
            else:
                self.qNetworkSlow.load_state_dict(
                    torch.load(os.path.join(folder, f'{name}.qNetworkSlow')),
                    map_location = map_location)
                self.qNetworkFast.load_state_dict(
                    torch.load(os.path.join(folder, f'{name}.qNetworkFast')),
                    map_location = map_location)

            self.memory.load(folder, name)
        except Exception as e:
            raise type(e)( 
                'lib.agents.Agent_DQN.Agent_DQN.load - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return
