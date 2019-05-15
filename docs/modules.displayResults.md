# modules.displayResults package

## Submodules

## modules.displayResults.displayResults module


#### modules.displayResults.displayResults.doSomething(logger)
print a line

This function simply prints a single line


* **Parameters**

    **logger** (*{logging.Logger}*) – The logger used for logging error information



#### modules.displayResults.displayResults.main(logger, resultsDict)
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
