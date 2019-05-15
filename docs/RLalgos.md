# RLalgos module


#### RLalgos.importModules(logger, resultsDict)
import and execute required modules

This function is used for importing all the
modules as defined in the ../config/modules.json
file and executing the main function within it
if present. In error, it fails gracefully …


* **Parameters**

    **logger** (*{logging.Logger}*) – logger module for logging information



#### RLalgos.main(logger, resultsDict)
main program

This is the place where the entire program is going
to be generated.
