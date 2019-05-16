import gym, sys
import numpy as np
from collections import deque 
import itertools as it

class Env:
    '''A convinience function for generating episodes and memories
    
    This convinience class generates a context manager that can be
    used for generating a Gym environment. This is supposed to be a
    drop-in replacement for the Unity environment. This however
    differs from the Unity environment in that it needs the name of
    the environment as input. The other difference is that there is
    no such thing as `trainMode`. 
    '''

    def __init__(self, envName, showEnv=False):
        '''Initialize the environment
        
        This sets up the requirements that will later be used for generating
        the gym Environment. The gym environment can be used in a mode that 
        hides the plotting of the actuual environment. This may result in a
        significant boost in speed. 
        
        Arguments:
            envName {str} -- The name of the environment to be generated. This
                shoould be a valid name. In case the namme provided is not a 
                valid name, this is going to exis with an error. 
        
        Keyword Arguments:
            showEnv {bool} -- Set this to ``True`` if you want to view the 
                environment (default: {False})
        '''

        try:
            self.no_graphics = not showEnv
            self.envName     = envName
            self.states      = None
        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.__init__ - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])
        return

    def __enter__(self):
        '''generate a context manager
        
        This will actually generate the context manager and allow you use this 
        within a ``with`` statement. This is the function that actually 
        initialized the environment and maintains it, until it is needed. 

        The idea of multiplel agents within the gym enviroonments doesnt exists
        as it does in the Unity agents. However, we shall incoroporoate this idea
        within the gym environment so that a signgle action can takke place. 
        
        Returns:
            ``this`` -- Returns an instance of the same class
        '''

        try:
            self.env   = gym.make(self.envName)
            self.state = self.env.reset()

        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.__enter__ - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return self

    def reset(self):
        '''reset the environment before starting an episode
        
        Returns:
            status -- The current status after the reset
        '''

        try:
            self.state = self.env.reset()
        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.reset - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])
        
        return self.states

    def step(self, policy):
        '''advance one step by taking an action
        
        This function takes a policy function and generates an action 
        according to that particular policy. This results in the 
        advancement of the episode into a one step with the return 
        of the reward, and the next state along with any done 
        information. 
        
        Arguments:
            policy {function} -- This function takes a state vector and 
                returns an action vector. It is assumed that the policy 
                is the correct type of policy, and is capable if taking
                the right returning the right type of vector corresponding
                the the policy for the current environment. It does not 
                check for the validity of the policy function
        
        Returns:
            list -- This returns a list of tuples containing the tuple 
                ``(s_t, a_t, r_{t+1}, s_{t+1}, d)``. One tuple for each
                agent. Even for the case of a single agent, this is going
                to return a list of states
        '''

        try:
            results = []

            states  = np.array([self.state])

            action  = policy(states)[0].cpu().detach().numpy()
            #print('A'*30, action, self.env.env.action_space.sample(), self.env.env.action_space)
            if type(self.env.env.action_space.sample()) == int:
                action  = int(action[0])

            nextState, reward, done, info = self.env.step(action)

            results.append((self.state, action, reward, nextState, done))
            
            self.state = nextState
            
        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.step - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return results

    def episode(self, policy, maxSteps=None):
        '''generate data for an entire episode
        
        This function generates an entire episde. It plays the environment
        by first resetting it too the beginning, and then playing the game for 
        a given number of steps (or unless the game is terminated). It generates
        a set of list of tuplees, again one for each agent. Rememebr that even
        when the number of agents is 1, it will still return a list oof states.

        Arguments:
            policy {function} -- The function that takes the current state and 
                returns the action vector. 
        
        Keyword Arguments:
            maxSteps {int or None} -- The maximum number of steps that the agent is
                going to play the episode before the episode is terminated. (default: 
                {None} in which case the episode will continue until it actually 
                finishes)
        
        Returns:
            list -- This returns the list of tuples for the entire episode. Again, this
                is a lsit of lists, one for each agent.
        '''

        try:
            self.reset()
            stepCount  = 0
            allResults = [[] for _ in range(1)] # One for each agent.

            while True:

                stepCount += 1
                result    = self.step(policy)[0]
                
                if not self.no_graphics:
                    self.env.render()

                state, action, reward, next_state, done = result
                allResults[0].append(result)

                if done:
                    break

                if (maxSteps is not None) and (stepCount >= maxSteps):
                    break

        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.episode - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return allResults

    def __exit__(self, exc, value, traceback):
        '''Exit the context manager
        
        The exit funciton that will result in exiting the
        context manager. Typically one is supposed to check 
        the error if any at this point. This will be handled 
        at a higher level
        
        Arguments:
            *args {[type]} -- [description]
        '''

        if not exec:
            self.env.close()
            return True

class Env1D:
    '''A convinience function for generating episodes and memories
    
    This convinience class generates a context manager that can be
    used for generating a Gym environment. This is supposed to be a
    drop-in replacement for the Unity environment. This however
    differs from the Unity environment in that it needs the name of
    the environment as input. The other difference is that there is
    no such thing as `trainMode`. 

    This 1D environment is designed to takke 1D state vector and use
    this vector in its calculations. If you are using a 1D environment
    you are advised to use this. 
    
    This environment has the added advantage that it will automatically
    stack together ``N`` previous states into a single state. Note that
    the first state will be copied ``N`` times, rather than zero padding
    as this seems a more natural state for the beginning. 
    '''

    def __init__(self, envName, N=1, showEnv=False):
        '''Initialize the environment
        
        This sets up the requirements that will later be used for generating
        the gym Environment. The gym environment can be used in a mode that 
        hides the plotting of the actuual environment. This may result in a
        significant boost in speed. 
        
        Arguments:
            envName {str} -- The name of the environment to be generated. This
                shoould be a valid name. In case the namme provided is not a 
                valid name, this is going to exis with an error. 
        
        Keyword Arguments:
            N {integer} -- Set this to the number of states that you wish to
            have that will be concatenated together. (default: 1). You will not
            be able to set a value less than 1.
            showEnv {bool} -- Set this to ``True`` if you want to view the 
                environment (default: {False})
            
        '''

        try:
            self.N           = N
            self.no_graphics = not showEnv
            self.envName     = envName
            self.states      = None

            assert type(self.N) == int, f'integer expected. Received {type(self.N)}'
            assert self.N > 0, f'self.N = {self.N} (should be greater than 0)'

        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.__init__ - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])
        return

    def __enter__(self):
        '''generate a context manager
        
        This will actually generate the context manager and allow you use this 
        within a ``with`` statement. This is the function that actually 
        initialized the environment and maintains it, until it is needed. 

        The idea of multiplel agents within the gym enviroonments doesnt exists
        as it does in the Unity agents. However, we shall incoroporoate this idea
        within the gym environment so that a signgle action can takke place. 
        
        Returns:
            ``this`` -- Returns an instance of the same class
        '''

        try:
            self.env   = gym.make(self.envName)
            state = self.env.reset()
            self.state = deque([state for i in range(self.N+1)], maxlen=self.N+1)

        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.__enter__ - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return self

    def reset(self):
        '''reset the environment before starting an episode
        
        Returns:
            status -- The current status after the reset
        '''

        try:
            state = self.env.reset()
            self.state = deque([state for i in range(self.N+1)], maxlen=self.N+1)

        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.reset - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])
        
        return self.states

    def step(self, policy):
        '''advance one step by taking an action
        
        This function takes a policy function and generates an action 
        according to that particular policy. This results in the 
        advancement of the episode into a one step with the return 
        of the reward, and the next state along with any done 
        information. 
        
        Arguments:
            policy {function} -- This function takes a state vector and 
                returns an action vector. It is assumed that the policy 
                is the correct type of policy, and is capable if taking
                the right returning the right type of vector corresponding
                the the policy for the current environment. It does not 
                check for the validity of the policy function
        
        Returns:
            list -- This returns a list of tuples containing the tuple 
                ``(s_t, a_t, r_{t+1}, s_{t+1}, d)``. One tuple for each
                agent. Even for the case of a single agent, this is going
                to return a list of states
        '''

        try:
            results = []

            state  = np.array(list(it.islice(self.state, 1, 1+self.N)))
            state  = state.flatten()
            states = np.array([state])

            action  = policy(states)[0].cpu().detach().numpy()
            #print('A'*30, action, self.env.env.action_space.sample(), self.env.env.action_space)
            if type(self.env.env.action_space.sample()) == int:
                action  = int(action[0])

            nextState, reward, done, info = self.env.step(action)
            self.state.append(nextState)

            nextState = np.array(list(it.islice(self.state, 1, 1+self.N)))
            nextState = nextState.flatten()
            
            results.append((state, action, reward, nextState, done))
            
        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.step - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return results

    def episode(self, policy, maxSteps=None):
        '''generate data for an entire episode
        
        This function generates an entire episde. It plays the environment
        by first resetting it too the beginning, and then playing the game for 
        a given number of steps (or unless the game is terminated). It generates
        a set of list of tuplees, again one for each agent. Rememebr that even
        when the number of agents is 1, it will still return a list oof states.

        Arguments:
            policy {function} -- The function that takes the current state and 
                returns the action vector. 
        
        Keyword Arguments:
            maxSteps {int or None} -- The maximum number of steps that the agent is
                going to play the episode before the episode is terminated. (default: 
                {None} in which case the episode will continue until it actually 
                finishes)
        
        Returns:
            list -- This returns the list of tuples for the entire episode. Again, this
                is a lsit of lists, one for each agent.
        '''

        try:
            self.reset()
            stepCount  = 0
            allResults = [[] for _ in range(1)] # One for each agent.

            while True:

                stepCount += 1
                result    = self.step(policy)[0]
                
                if not self.no_graphics:
                    self.env.render()

                state, action, reward, next_state, done = result
                allResults[0].append(result)

                if done:
                    break

                if (maxSteps is not None) and (stepCount >= maxSteps):
                    break

        except Exception as e:
            raise type(e)( 
                'lib.envs.envUnity.Env.episode - ERROR - ' + str(e) 
                ).with_traceback(sys.exc_info()[2])

        return allResults

    def __exit__(self, exc, value, traceback):
        '''Exit the context manager
        
        The exit funciton that will result in exiting the
        context manager. Typically one is supposed to check 
        the error if any at this point. This will be handled 
        at a higher level
        
        Arguments:
            *args {[type]} -- [description]
        '''

        if not exec:
            self.env.close()
            return True

