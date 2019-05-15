# modules.tests package

## Submodules

## modules.tests.testActors module


#### modules.tests.testActors.allTests(logger)

#### modules.tests.testActors.testArgMax(logger)
## modules.tests.testCritics module


#### modules.tests.testCritics.allTests(logger)
## modules.tests.testMemoryBuffers module


#### modules.tests.testMemoryBuffers.allTests(logger)
## modules.tests.testOpenAI module


#### modules.tests.testOpenAI.allTests(logger)

#### modules.tests.testOpenAI.checkValidity(logger, name)

#### modules.tests.testOpenAI.playOne(logger, name)

#### modules.tests.testOpenAI.renderOne(logger, name, sleepTime)
## modules.tests.testPolicy module


#### modules.tests.testPolicy.allTests(logger)
## modules.tests.testQnetwork module


#### modules.tests.testQnetwork.allTests(logger)

#### modules.tests.testQnetwork.testQnetwork(logger)
## modules.tests.testUnity module


#### modules.tests.testUnity.allTests(logger)
## modules.tests.tests module


#### modules.tests.tests.main(logger, resultsDict)
main function for module1

This function finishes all the tasks for the
main function. This is a way in which a
particular module is going to be executed.


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **resultsDict** (*{dict}*) – A dintionary containing information about the
      command line arguments. These can be used for
      overwriting command line arguments as needed.


## Module contents

[one line description of the module]

[this is a
multiline description of what the module does.]

### Before you Begin

Make sure that the configuration files are properly set, as mentioned in the Specifcations
section. Also, [add any other housekeeping that needs to be done before starting the module].

### Details of Operation

[
Over here, you should provide as much information as possible for what the modules does.
You should mention the data sources that the module uses, and important operations that
the module performs.
]

### Results

[
You want to describe the results of running this module. This would include instances of
the database that the module updates, as well as any other files that the module creates.
]

### Specifications:

Specifications for running the module is described below. Note that all the json files
unless otherwise specified will be placed in the folder `config` in the main project
folder.

#### Specifications for the database:

[
Note the tables within the various databases that will be affected by this module.
]

#### Specifications for `modules.json`

Make sure that the `execute` statement within the modules file is set to True.

```
"moduleName" : "module1",
"path"       : "modules/module1/module1.py",
"execute"    : true,
"description": "",
"owner"      : ""
```

#### Specification for [any other files]

[
Make sure that you specify all the other files whose parameters will need to be
changed.
]
