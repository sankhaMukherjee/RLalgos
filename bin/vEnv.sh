#!/bin/bash

#----------------------------------------------
# Note that this is the standard way of doing 
# things in Python 3.6. Earlier versions used
# virtulalenv. It is best to convert to 3.6
# before you do anything else. 
# Note that the default Python version in 
# the AWS Ubuntu is 3.5 at the moment. You
# will need to upgrade the the new version 
# if you wish to use this environment in 
# AWS
#----------------------------------------------
python3 -m venv env

# this is for bash. Activate
# it differently for different shells
#--------------------------------------
source env/bin/activate 

pip3 install --upgrade pip

if [ -e requirements.txt ]; then

    pip3 install -r requirements.txt

else

    pip3 install pytest
    pip3 install pytest-cov
    pip3 install sphinx
    pip3 install sphinx_rtd_theme

    # Logging into logstash
    pip3 install python-logstash

    # networkX for graphics
    pip3 install networkx
    pip3 install pydot # dot layout
    
    # Utilities
    pip3 install jupyter
    pip3 install tqdm

    # scientific libraries
    pip3 install numpy
    pip3 install scipy 
    pip3 install tqdm

    pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
    pip3 install torchvision

    #########################################################
    # Allows us to run things form precompiled binaries
    #  from UDACITY. Obtained the python folder from 
    #  UDACITY as well. This seems to work well with 
    #  their own preecompiled binaries which I have also
    #  downloaded. 
    # Does not allow us to run the more recent binaries
    # I have not been able to generate proper binaries,
    #  howoever, the python scripts seem to be able to
    #  load them.
    #########################################################
    if [ ! -e installers ]; then
        mkdir installers
    fi
    cd installers
    pip3 install -e ./python 

    #########################################################
    # This allows practically all the Atari games as well
    # as all the simple games. Only paid stuff is not 
    # included here.
    #########################################################
    if [ ! -e gym ]; then
        git clone https://github.com/openai/gym
    fi

    pip3 install -e gym
    pip3 install -e gym[atari]

    pip3 freeze > requirements.txt

fi

deactivate