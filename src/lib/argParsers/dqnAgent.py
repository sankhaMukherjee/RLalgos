from logs import logDecorator as lD
import json

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.argParsers.dqnAgent'

@lD.log(logBase + '.parsersAdd')
def addParsers(logger, parser):
    '''add argument parsers specific to the ``config/config.json`` file
    
    This function is kgoing to add argument parsers specific to the 
    ``config/config.json`` file. This file has several options for 
    logging data. This information will be 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    parser : {argparse.ArgumentParser instance}
        An instance of ``argparse.ArgumentParser()`` that will be
        used for parsing the command line arguments specific to the 
        config file
    
    Returns
    -------
    argparse.ArgumentParser instance
        The same parser argument to which new CLI arguments have been
        appended
    '''
    
    parser.add_argument("--dqnAgent_nIterations",
        type = int,
        help="number of iterations that the agent should train for")

    parser.add_argument("--dqnAgent_initMemoryIterations",
        type = int,
        help="number of iterations that the initial memory should use for generating some initial memory")

    parser.add_argument("--dqnAgent_eps0",
        type = float,
        help="initial epsilon for an epsilon greedy policy")

    parser.add_argument("--dqnAgent_epsDecay",
        type = float,
        help="decay factor that the epsilon should be multiplying by to get a particular rate of epsilon decay")

    parser.add_argument("--dqnAgent_minEps",
        type = float,
        help="minimum epsilon that the learner should maintain for the purpose of learning")

    parser.add_argument("--dqnAgent_maxSteps",
        type = int,
        help="max number of steps that the agent should plat at a maximum for an episode")

    parser.add_argument("--dqnAgent_nSamples",
        type = int,
        help="number of sampes that the DQN agent should take for a single training pass")

    parser.add_argument("--dqnAgent_Tau",
        type = float,
        help="rate by which the slow learner should update its weight for learning purposes")

    parser.add_argument("--dqnAgent_lr",
        type = float,
        help="the learning rate of the q network")

    return parser

@lD.log(logBase + '.decodeParser')
def decodeParser(logger, args):
    '''generate a dictionary from the parsed args
    
    The parsed args may/may not be present. When they are
    present, they are pretty hard to use. For this reason,
    this function is going to convert the result into
    something meaningful.
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    args : {args Namespace}
        parsed arguments from the command line
    
    Returns
    -------
    dict
        Dictionary that converts the arguments into something
        meaningful
    '''


    values = {}

    try:
        if args.dqnAgent_nIterations is not None:
            values["nIterations"] = args.dqnAgent_nIterations
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_nIterations: {e}')
    try:
        if args.dqnAgent_initMemoryIterations is not None:
            values["initMemoryIterations"] = args.dqnAgent_initMemoryIterations
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_initMemoryIterations: {e}')
    try:
        if args.dqnAgent_eps0 is not None:
            values["eps0"] = args.dqnAgent_eps0
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_eps0: {e}')
    try:
        if args.dqnAgent_epsDecay is not None:
            values["epsDecay"] = args.dqnAgent_epsDecay
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_epsDecay: {e}')
    try:
        if args.dqnAgent_minEps is not None:
            values["minEps"] = args.dqnAgent_minEps
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_minEps: {e}')
    try:
        if args.dqnAgent_maxSteps is not None:
            values["maxSteps"] = args.dqnAgent_maxSteps
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_maxSteps: {e}')
    try:
        if args.dqnAgent_nSamples is not None:
            values["nSamples"] = args.dqnAgent_nSamples
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_nSamples: {e}')
    try:
        if args.dqnAgent_Tau is not None:
            values["Tau"] = args.dqnAgent_Tau
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_Tau: {e}')

    try:
        if args.dqnAgent_Tau is not None:
            values["lr"] = args.dqnAgent_lr
    except Exception as e:
        logger.error(f'Unable to decode the value for args.dqnAgent_lr: {e}')
    
    
    return values

