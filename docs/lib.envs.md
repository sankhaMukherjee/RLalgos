# lib.envs package

## Submodules

## lib.envs.envDiscAction module


#### class lib.envs.envDiscAction.Env(fileName, showEnv=False, trainMode=True)
Bases: `object`

A convinience function for generating episodes and memories

This convinience class generates a context manager that can be
used for generating a simple discrete environment. This is supposed
to be a drop-in replacement for the different for any other
environment. This environment is useful for testing whether a
an Agent that has too select discrete actions is properly doing
its job. This does not take any input parameters, and reeturns
only the the single environment. The environment is shown below.


#### __enter__()
generate a context manager

This will actually generate the context manager and allow you use this
within a `with` statement. This is the function that actually
initialized the environment and maintains it, until it is needed.


* **Returns**

    `this` – Returns an instance of the same class



#### __exit__(exc, value, traceback)
Exit the context manager

The exit funciton that will result in exiting the
context manager. Typically one is supposed to check
the error if any at this point. This will be handled
at a higher level


* **Parameters**

    **{****[****type****]****} --**** [****description****]** (*\*args*) – 



#### episode(policy, maxSteps=None)
generate data for an entire episode

This function generates an entire episde. It plays the environment
by first resetting it too the beginning, and then playing the game for
a given number of steps (or unless the game is terminated). It generates
a set of list of tuplees, again one for each agent. Rememebr that even
when the number of agents is 1, it will still return a list oof states.


* **Parameters**

    **{function} -- The function that takes the current state and** (*policy*) – returns the action vector.



* **Keyword Arguments**

    **{int**** or ****None} -- The maximum number of steps that the agent is** (*maxSteps*) – going to play the episode before the episode is terminated. (default:
    {None} in which case the episode will continue until it actually
    finishes)



* **Returns**

    list – This returns the list of tuples for the entire episode. Again, this

        is a lsit of lists, one for each agent.




#### reset()
reset the environment before starting an episode


* **Returns**

    status – The current status after the reset



#### step(policy)
advance one step by taking an action

This function takes a policy function and generates an action
according to that particular policy. This results in the
advancement of the episode into a one step with the return
of the reward, and the next state along with any done
information.


* **Parameters**

    **{function} -- This function takes a state vector and** (*policy*) – returns an action vector. It is assumed that the policy
    is the correct type of policy, and is capable if taking
    the right returning the right type of vector corresponding
    the the policy for the current environment. It does not
    check for the validity of the policy function



* **Returns**

    list – This returns a list of tuples containing the tuple

        `(s_t, a_t, r_{t+1}, s_{t+1}, d)`. One tuple for each
        agent. Even for the case of a single agent, this is going
        to return a list of states



## lib.envs.envGym module


#### class lib.envs.envGym.Env(envName, showEnv=False)
Bases: `object`

A convinience function for generating episodes and memories

This convinience class generates a context manager that can be
used for generating a Gym environment. This is supposed to be a
drop-in replacement for the Unity environment. This however
differs from the Unity environment in that it needs the name of
the environment as input. The other difference is that there is
no such thing as trainMode.


#### __enter__()
generate a context manager

This will actually generate the context manager and allow you use this
within a `with` statement. This is the function that actually
initialized the environment and maintains it, until it is needed.

The idea of multiplel agents within the gym enviroonments doesnt exists
as it does in the Unity agents. However, we shall incoroporoate this idea
within the gym environment so that a signgle action can takke place.


* **Returns**

    `this` – Returns an instance of the same class



#### __exit__(exc, value, traceback)
Exit the context manager

The exit funciton that will result in exiting the
context manager. Typically one is supposed to check
the error if any at this point. This will be handled
at a higher level


* **Parameters**

    **{****[****type****]****} --**** [****description****]** (*\*args*) – 



#### episode(policy, maxSteps=None)
generate data for an entire episode

This function generates an entire episde. It plays the environment
by first resetting it too the beginning, and then playing the game for
a given number of steps (or unless the game is terminated). It generates
a set of list of tuplees, again one for each agent. Rememebr that even
when the number of agents is 1, it will still return a list oof states.


* **Parameters**

    **{function} -- The function that takes the current state and** (*policy*) – returns the action vector.



* **Keyword Arguments**

    **{int**** or ****None} -- The maximum number of steps that the agent is** (*maxSteps*) – going to play the episode before the episode is terminated. (default:
    {None} in which case the episode will continue until it actually
    finishes)



* **Returns**

    list – This returns the list of tuples for the entire episode. Again, this

        is a lsit of lists, one for each agent.




#### reset()
reset the environment before starting an episode


* **Returns**

    status – The current status after the reset



#### step(policy)
advance one step by taking an action

This function takes a policy function and generates an action
according to that particular policy. This results in the
advancement of the episode into a one step with the return
of the reward, and the next state along with any done
information.


* **Parameters**

    **{function} -- This function takes a state vector and** (*policy*) – returns an action vector. It is assumed that the policy
    is the correct type of policy, and is capable if taking
    the right returning the right type of vector corresponding
    the the policy for the current environment. It does not
    check for the validity of the policy function



* **Returns**

    list – This returns a list of tuples containing the tuple

        `(s_t, a_t, r_{t+1}, s_{t+1}, d)`. One tuple for each
        agent. Even for the case of a single agent, this is going
        to return a list of states



## lib.envs.envUnity module


#### class lib.envs.envUnity.Env(fileName, showEnv=False, trainMode=True)
Bases: `object`

A convinience function for generating episodes and memories

This convinience class generates a context manager that can be
used for generating a Unity environment. The Unity environment
and the OpenAI Gym environment operates slightly differently
and hence it will be difficult to create a uniform algorithm that
is able to solve everything at the sametime. This environment
tries to solve that problem.


#### __enter__()
generate a context manager

This will actually generate the context manager and allow you use this
within a `with` statement. This is the function that actually
initialized the environment and maintains it, until it is needed.


* **Returns**

    `this` – Returns an instance of the same class



#### __exit__(exc, value, traceback)
Exit the context manager

The exit funciton that will result in exiting the
context manager. Typically one is supposed to check
the error if any at this point. This will be handled
at a higher level


* **Parameters**

    **{****[****type****]****} --**** [****description****]** (*\*args*) – 



#### episode(policy, maxSteps=None)
generate data for an entire episode

This function generates an entire episde. It plays the environment
by first resetting it too the beginning, and then playing the game for
a given number of steps (or unless the game is terminated). It generates
a set of list of tuplees, again one for each agent. Rememebr that even
when the number of agents is 1, it will still return a list oof states.


* **Parameters**

    **{function} -- The function that takes the current state and** (*policy*) – returns the action vector.



* **Keyword Arguments**

    **{int**** or ****None} -- The maximum number of steps that the agent is** (*maxSteps*) – going to play the episode before the episode is terminated. (default:
    {None} in which case the episode will continue until it actually
    finishes)



* **Returns**

    list – This returns the list of tuples for the entire episode. Again, this

        is a lsit of lists, one for each agent.




#### reset()
reset the environment before starting an episode


* **Returns**

    status – The current status after the reset



#### step(policy)
advance one step by taking an action

This function takes a policy function and generates an action
according to that particular policy. This results in the
advancement of the episode into a one step with the return
of the reward, and the next state along with any done
information.


* **Parameters**

    **{function} -- This function takes a state vector and** (*policy*) – returns an action vector. It is assumed that the policy
    is the correct type of policy, and is capable if taking
    the right returning the right type of vector corresponding
    the the policy for the current environment. It does not
    check for the validity of the policy function



* **Returns**

    list – This returns a list of tuples containing the tuple

        `(s_t, a_t, r_{t+1}, s_{t+1}, d)`. One tuple for each
        agent. Even for the case of a single agent, this is going
        to return a list of states



## Module contents

several environments are available for immediate import

This library contains containerized versions of the different
environments that can be used for training a number of
environments. This is essential for being able to checke the
qualit of the different learning algorithms. Currently the different
environments available are as foollows:

> * envUnity: The Unity Environment

> * envGym: The gym environment

The details of instaling each of these environments will be shown
below …
