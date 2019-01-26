from collections import deque
import numpy as np
import pickle, os

class SimpleReplayBuffer:

    def __init__(self, maxDataTuples):
        '''The replay buffer
        
        Save data for the replay buffer
        
        Parameters
        ----------
        maxDataTuples : {int}
            The size of the ``deque`` that is used for storing the
            data tuples. This assumes that the data tuples are 
            present in the form: ``(state, action, reward, next_state, 
            done, cumRewards)``. This means that we assume that the 
            data will have some form of cumulative reward pints associated
            with each tuple.
        '''
        self.maxDataTuples  = maxDataTuples
        self.memory         = deque(maxlen=maxDataTuples)
        return

    def append(self, result):
        '''append a single tuple to the current replay buffer
        
        This function allows someone to add a single tuple to
        the replay buffer. 
        
        Parameters
        ----------
        result : {tuple}
            The tuple that should be added into the memory buffer. 
        '''
        self.memory.append(result)
        return

    @property
    def len(self):
        '''returns the length of the emory buffer
        
        Remember that this is a property and there is no need
        to call it as a function.

        Returns:
            int -- the length of the currrent memory buffer
        '''
        return len(self.memory)

    @property
    def shape(self):
        '''the shape of the buffer
        
        This is the shape of the memory buffers. This returns
        a tuple that contains the length of the buffer for the
        first element of the tuple, and the length of each element 
        as the second element of the tuple. If there is nothing 
        within the memory, this is going to return a None
        
        Returns:
            tuple -- the shape of the data within the memory buffer
        '''
        N = len(self.memory)
        if N > 0:
            N1 = len( self.memory[0] )
        else:
            N1 = None
        return N, N1

    def appendMany(self, results):
        '''append multiple tuples to the memory buffer
        
        Most often we will not be insterested in inserting a single data point
        into the replay buffer, but rather a whole list of these. This function
        just iterates over this list and inserts each tuple one by one.
        
        Parameters
        ----------
        results : {list}
            List of tuples that are to be inserted into the replay buffer.
        '''
        for r in results:
            self.memory.append(r)
        return

    def appendAllAgentResults(self, allResults):
        '''append all data from all agents into the same buffer
        
        This is useful when there is only one agent or when all the agents represent
        the same exact larning characteristics. In this case, multiple agents can be
        simulated by the same function.
        
        Arguments:
            allResults {list} -- List of list tuples to be entered into the buffer. 
        '''

        for results in allResults:
            self.appendMany( results )

        return

    def sample(self, nSamples):
        '''sample from the replay beffer
        
        This function samples form the memory buffer, and returns the number of
        samples required. This does sampling in an intelligent manner. Since we are
        saving the cumulative rewards, we selectively select values that provide
        us greater 
        
        Parameters
        ----------
        nSamples : {int}
            The number of memory elements to return
        
        Returns
        -------
        list
            A list of samples that can be used for sampling the data. 
        '''

        choice = np.random.choice( np.arange( len(self.memory) ), nSamples)
        results = [ self.memory[c] for c in choice]
        
        return results

    def save(self, folder, name):
        '''save the replay buffer
        
        This function is going to save the data within the replay buffer
        into a pickle file. This will allow us to reload the buffer to 
        a state where it has already been saved.
        
        Parameters
        ----------
        folder : {str}
            path to the folder where the data is to be saved
        name : {str}
            Name associated with the buffer. Since this program has two agents
            acting in tandum, we need to provide a name that will identify which
            agent's buffer we are saving. 
        '''

        with open(os.path.join(folder, f'memory_{name}.pickle'), 'wb') as fOut:
            pickle.dump(self.memory, fOut, pickle.HIGHEST_PROTOCOL)

        return

    def load(self, folder, name):
        '''load the data from a particular file
        
        Data saved with the previous command can be reloaded into this new buffer.
        
        Parameters
        ----------
        folder : {str}
            Path to the folder where the data is saved
        name : {str}
            Name of the agent associated whose data is to be extracted.
        '''
        self.memory = pickle.load(open( os.path.join(folder, f'memory_{name}.pickle'), 'rb' ))
        return