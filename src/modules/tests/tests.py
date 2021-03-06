from logs import logDecorator as lD 
import json, pprint

from modules.tests import testOpenAI         as tOAI
from modules.tests import testUnity          as tUnity
from modules.tests import testMemoryBuffers  as tMB
from modules.tests import testPolicy         as tP
from modules.tests import testActors         as tA
from modules.tests import testCritics        as tC
from modules.tests import testQnetwork       as tQn

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.tests.tests'


@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''


    try:

        cfg  = json.load(open('../config/modules/tests.json'))['params']

        if cfg['TODO']['openAI']:
            tOAI.allTests()
            
        if cfg['TODO']['Unity']:
            tUnity.allTests()

        if cfg['TODO']['MemoryBuffer']:
            tMB.allTests()

        if cfg['TODO']['policy']:
            tP.allTests()
        
        if cfg['TODO']['actors']:
            tA.allTests()
        
        if cfg['TODO']['critics']:
            tC.allTests()

        if cfg['TODO']['qNetwork']:
            tQn.allTests()

    except Exception as e:
        logger.error(f'Unable to complete all the tests: {e}')

    return

