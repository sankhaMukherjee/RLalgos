from unityagents import UnityEnvironment
import numpy as np
import json

from collections import deque, namedtuple
from tqdm import tqdm

class Env:
    '''A convinience function for generating episodes and memories
    
    This convinience class generates a context manager that can be
    used for generating a Unity environment. The Unity environment
    and the OpenAI Gym environment operates slightly differently
    and hence it will be difficult to create a uniform algorithm that
    is able to solve everything at the sametime. This environment
    tries to solve that problem.
    '''

    def __init__(self, fileName, showEnv=False, trainMode=True):
        '''Initialize the environment
        
        This sets up the requirements that will later be used for generating
        the Unity Environment. This assumes that you will provide a binary
        file for generating the environment. There are different ways in 
        which the environment can be generated. It can be generated either
        in a *headless* mode by using showEnv as False, in which case the 
        environment will not show a window at startup. This is good for 
        training, as well as situations when you are running the environment
        without the presence of an X server, especially when you are running 
        this environment remotely. The other thing that you can do is to 
        specify that this is being run in `trainMode`. In this case, the 
        environment will be primed for training. That is, each frame will
        finish as soon as possible. This is not good for observing what is
        happening. However, this significantly increases the speed of 
        training. 
        
        Arguments:
            fileName {str} -- Path to the binary file. This file must be
                the same as the one for which the `unityagents` package 
                has been generated. 
        
        Keyword Arguments:
            showEnv {bool} -- Set this to ``True`` if you want to view the 
                environment (default: {False})
            trainMode {bool} -- Set this to ``True`` if you want the environment
                tobe in training mode (i.e. fast execution) (default: {True})
        '''

        try:
            self.no_graphics = not showEnv
            self.trainMode   = trainMode
            self.fileName    = fileName
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
        
        Returns:
            ``this`` -- Returns an instance of the same class
        '''

        try:
            self.env    = UnityEnvironment(
                file_name   = self.fileName, 
                no_graphics = self.no_graphics )

            # get the default brain
            self.brain_name = self.env.brain_names[0]
            self.brain      = self.env.brains[self.brain_name]
            self.env_info   = self.env.reset(train_mode = self.trainMode)[self.brain_name]

            self.num_agents  = len(self.env_info.agents)
            self.action_size = self.brain.vector_action_space_size
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
            self.env.reset(train_mode=self.trainMode)
            self.states = self.env_info.vector_observations
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
            states      = self.states.copy()
            actions     = policy(states)
            env_info    = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations 
            rewards     = env_info.rewards             
            dones       = env_info.local_done          

            self.states = next_states

            results = []
            for i in range(self.num_agents):
                state       = states[i]
                action      = actions[i]
                reward      = rewards[i]
                next_state  = next_states[i]
                done        = dones[i]

                results.append((state, action, reward, next_state, done))

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
            stepCount     = 0
            allResults    = [[] for _ in range(self.num_agents)]

            while True:

                stepCount += 1
                finished  = False
                results   = self.step(policy)
                for agent in range(self.num_agents):
                    state, action, reward, next_state, done = results[agent]
                    allResults[agent].append(results[agent])
                    finished = finished or done

                if finished:
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

