'''module that contains a plethora of agents and policies

This module will contain a number of agents and policies that can be specified 
by easily in a uniform manner. This way different agents can be easily swapped 
out between each other. 

Agents
-------

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

 - ``act``     : action to take given a state
 - ``save``    : save the current state of the agent
 - ``restore`` : restore the agent from a state saved earlier

Currently the following agents are available:


Policies
---------

Policies determine what action to take given the result of an agent. In one
way, they are similar in that they also take a state, and return an action. 
However, a policy determines how much exploration vs. exploitation should
be done over the period that the agent is learning.

'''