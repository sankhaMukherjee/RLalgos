# RLalgos

A testbed for finding the efficacy of a lot of different types of reinforcement algorithms

## Getting Started

There are esentially two different types of deployment that you can use. The openAI
gym environment, which contains a ton of Atari games, and some Unity games that can
be run using precompiled binaries. Curreently the implementation uses procompiled binaries
from the Reinforcement Learning course at Udacity. Hence, I have had to use their other 
libraries. I have included these within this repo so that you will not have to downloaod
them again. However, if you wish, you may download them form the origiinal location over 
[here](https://github.com/udacity/deep-reinforcement-learning/tree/master)

Follow the installation instruction in the installation section. After that, read the documentation
available [here](). This is designed to be very modular, and has a lot of different enviroonments. 
Hence, it would be meaningful to start slow and read through the documentation slowly. 

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

## Installing

The folloiwing installations are for \*nix-like systems. This is currently tested in the following system: `Ubuntu 18.10`. 

For installation, first close this repository, and generate the virtual environment required for running the programs. 

This project framework uses [venv](https://docs.python.org/3/library/venv.html) for maintaining virtual environments. Please familiarize yourself with [venv](https://docs.python.org/3/library/venv.html) before working with this repository. You do not want to contaminate your system python while working with this repository.

A convenient script for doing this is present in the file [`bin/vEnv.sh`](./bin/vEnv.sh). This is automatically do the following things:

1. Generate a virtual environment
2. activate this environment
3. install all required libraries
4. deactivate the virtual environment and return to the prompt. 

At this point you are ready to run programs. However, remember that you will need to activate the virtual environment any time you use the program.

For activating your virtual environment, type `source env/bin/activate` in the project folder in [bash](https://www.gnu.org/software/bash/) or `source env/bin/activate.fish` if you are using the [fish](https://fishshell.com/) shell.
For deactivating, just type deactivate in your shell

** Note: The python file contains a modified version of the python folder available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python). This has been done to make sure that the program works properly with my current system configuration. This may change in the future. 

## Contributing

Please send in a pull request.

## Authors

Sankha S. Mukherjee - Initial work (2019)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

 - Practically all of the implementations are a result of the UDACITY Deep Reinforcement Learning Course
 - Hopefully I shall be able to expand to more real-world problems
 - etc.
 
