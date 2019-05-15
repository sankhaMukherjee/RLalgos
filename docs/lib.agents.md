# lib.agents package

## Submodules

## lib.agents.Agent_DQN module


#### class lib.agents.Agent_DQN.Agent_DQN(env, memory, qNetworkSlow, qNetworkFast, numActions, gamma, device='cpu')
Bases: `object`


#### checkTrainingMode()
[summary]

[description]


#### epsGreedyAction(state, eps=0.999)
epsilon greedy action

This is the epsilon greedy action. In general, this is going to
select the maximum action `eps` percentage of the times, while
selecting the random action the rest of the time. It is assumed
that this will receive a value of epsilon between 0 and 1.


* **Parameters**

    * **state** (*{ndarray}*) – [description]

    * **eps** (*float**, **optional*) – Determines the fraction of times the max action will be selected
      in comparison to a random action. (the default is 0.999)



* **Returns**

    The 1d tensor that has an action for each state provided.



* **Return type**

    tensor



#### eval()
[summary]

[description]


#### load(folder, name, map_location=None)
load the model

An agent saved with the save command can be safely loaded with this command.
This will load both the qNetworks, as well as the memory buffer. There is a
possibility that one may not want to load the model into the same device. In
that case, the user should insert the device that the user wants to load the
model into.


* **Parameters**

    * **folder** (*{str}*) – folder into which the model should be saved.

    * **name** (*{str}*) – A name to associate the model to load. It is absolutelty possible to save
      a number of models within the same folder, and hence the name can retrieve
      that model that is important.

    * **map_location** (*{str}**, **optional*) – The device in which to load the file. This is a string like ‘cpu’, ‘cuad:0’
      etc. (the default is None, which results in the model being loaded to the
      originam device)



#### maxAction(state)
returns the action that maximizes the Q function

Given an set of statees, this function is going to return a set
of actions which will maximize the value of the Q network for each
of the supplied states.


* **Parameters**

    **state** (*{nd_array** or **tensor}*) – numpy array or tensor containing the state. The columns
    represent the different parts of the state.



* **Returns**

    The return values of actions that maximize the states



* **Return type**

    uarray



#### memoryUpdateEpisode(policy, maxSteps=1000, minScoreToAdd=None)
update the memory

Given a particular policy, this memory is going to take
the policy and generate a series of memories and update
thememory buffer. Generating memories is easier to do
using this function than an external function …


* **Parameters**

    * **policy** (*{function}*) – This is a function that takes a state and returns an action. This
      defines how the agent will explore the environment by changing the
      exploration/exploitation scale.

    * **maxSteps** (*{number}**, **optional*) – The maximum number of steps that one shoule have within an episode.
      (the default is 1000)



#### randomAction(state)
returns a set of random actions for the given states

given the size of the number of actions, this function is going
to return a set of actions that has the same number of actions
as the number of inputs in the shape. For example, if
`state.shape == (10, ?)` then the result will be a vector of
size 10. This is in accordance with the redduction in the
dimensionality of the maxAction space.


* **Parameters**

    **state** (*{nd_array** or **tensor}*) – numpy array or tensor containing the state. The columns
    represent the different parts of the state.



* **Returns**

    The return value is set of random actions



* **Return type**

    uarray



#### save(folder, name)
save the model

This function allows one to save the model, in a folder that is
specified, with the fast and the slow qNetworks, as well as the
memory buffer. Sometimes there may be more than a single agent,
and under those circumstances, the name will come in handy. If the
supplied folder does not exist, it will be generated.


* **Parameters**

    * **folder** (*{str}*) – folder into which the model should be saved.

    * **name** (*{str}*) – A name to associate the current model with. It is
      absolutelty possible to save a number of models within
      the same folder.



#### softUpdate(tau=0.1)
update the slow network slightly

This is going to update the slow network slightly. The amount
is dictated by `tau`. This should be a number between 0 and 1.
It will update the `tau` fraction of the slow network weights
with the new weights. This is done for providing stability to the
network.


* **Parameters**

    **tau** (*{number}**, **optional*) – This parameter determines how much of the fast Networks weights
    will be updated to the ne parameters weights (the default is 0.1)



#### step(nSamples=100)
## lib.agents.policy module


#### class lib.agents.policy.epsGreedyPolicy(agent, randomAgent)
Bases: `object`


#### act(states, eps)
[summary]

[description]


* **Parameters**

    * **{****[****type****]****} --**** [****description****]** (*eps*) – 

    * **{****[****type****]****} --**** [****description****]** – 



* **Returns**

    [type] – [description]


## lib.agents.qNetwork module


#### class lib.agents.qNetwork.qNetworkDiscrete(stateSize, actionSize, layers=[10, 5], activations=[<function tanh>, <function tanh>], batchNormalization=False, lr=0.01)
Bases: `torch.nn.modules.module.Module`


#### forward(x)
forward function that is called during the forward pass

This is the forward function that will be called during a
forward pass. It takes thee states and gives the Q value
correspondidng to each of the applied actions that are
associated with that state.


* **Parameters**

    **x** (*{tensor}*) – This is a 2D tensor.



* **Returns**

    This represents the Q value of the function



* **Return type**

    tensor



#### step(v1, v2)
[summary]

[description]


* **Parameters**

    * **v1** (*{**[**type**]**}*) – [description]

    * **v2** (*{**[**type**]**}*) – [description]



* **Raises**

    `type` – [description]


## lib.agents.randomActor module


#### class lib.agents.randomActor.randomDiscreteActor(stateShape, numActions)
Bases: `object`


#### act(state)
return an action based on the state

[description]


* **Parameters**

    **{nd-array} -- nd-array as described by the state** (*state*) – shape described in the `__init__` function.



* **Returns**

    integer – integer between 0 and the number of actions

        available.



## lib.agents.sequentialActor module


#### class lib.agents.sequentialActor.SequentialDiscreteActor(stateSize, numActions, layers=[10, 5], activations=[<function tanh>, <function tanh>], batchNormalization=True)
Bases: `torch.nn.modules.module.Module`


#### forward(x)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

## lib.agents.sequentialCritic module


#### class lib.agents.sequentialCritic.SequentialCritic(stateSize, actionSize, layers=[10, 5], activations=[<function tanh>, <function tanh>], mergeLayer=0, batchNormalization=True)
Bases: `torch.nn.modules.module.Module`


#### forward(x, action)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

## Module contents

module that contains a plethora of agents and policies

This module will contain a number of agents and policies that can be specified
by easily in a uniform manner. This way different agents can be easily swapped
out between each other.

### Agents

Agents take a state and return an action. Unfortunately, both states and
actions come in a variety of shapes and sizes.

There are essentially two different types of states. Either a vector,
or an nd-array. Typically, nd-arrays are dealt with coonvolution operators
whiile vectors are dealt with simple sequential networks. While nd-arrays
can be flattened, the reverse operator us generally not practicable. In
any case, it is assumed that the user has enough intuition to be able to
distinguish between the two.

Actions are typically vectors. However, sometimes actions can be discrete
and sometimes contnuous and any combination of the two. Furthermore, actions
typically have bounds. In more general cases (like chess) valid actions are
associated with the current state. There is no generic way of solving this
problem, so we chall create different types of agents that will return
different types of actions.

All agets will definitely have the following methods:

> * `act`     : action to take given a state

> * `save`    : save the current state of the agent

> * `restore` : restore the agent from a state saved earlier

### Policies

Policies determine what action to take given the result of an agent. In one
way, they are similar in that they also take a state, and return an action.
However, a policy determines how much exploration vs. exploitation should
be done over the period that the agent is learning.
