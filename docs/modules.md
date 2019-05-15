# modules package

## Subpackages

* modules.DQNagent package

  * Submodules

  * modules.DQNagent.DQNagent module

  * modules.DQNagent.runAgent module

  * Module contents

    * Before you Begin

    * Details of Operation

    * Results

    * Specifications:

      * Specifications for the database:

      * Specifications for `modules.json`

      * Specification for [any other files]

* modules.displayResults package

  * Submodules

  * modules.displayResults.displayResults module

  * Module contents

    * Before you Begin

    * Details of Operation

    * Results

    * Specifications:

      * Specifications for the database:

      * Specifications for `modules.json`

      * Specification for [any other files]

* modules.module1 package

  * Submodules

  * modules.module1.module1 module

  * Module contents

    * Before you Begin

    * Details of Operation

    * Results

    * Specifications:

      * Specifications for the database:

      * Specifications for `modules.json`

      * Specification for [any other files]

* modules.playAgent package

  * Submodules

  * modules.playAgent.playAgent module

  * Module contents

    * Before you Begin

    * Details of Operation

    * Results

    * Specifications:

      * Specifications for the database:

      * Specifications for `modules.json`

      * Specification for [any other files]

* modules.testAgents package

  * Submodules

  * modules.testAgents.testAgents module

  * Module contents

    * Before you Begin

    * Details of Operation

    * Results

    * Specifications:

      * Specifications for the database:

      * Specifications for `modules.json`

      * Specification for [any other files]

* modules.tests package

  * Submodules

  * modules.tests.testActors module

  * modules.tests.testCritics module

  * modules.tests.testMemoryBuffers module

  * modules.tests.testOpenAI module

  * modules.tests.testPolicy module

  * modules.tests.testQnetwork module

  * modules.tests.testUnity module

  * modules.tests.tests module

  * Module contents

    * Before you Begin

    * Details of Operation

    * Results

    * Specifications:

      * Specifications for the database:

      * Specifications for `modules.json`

      * Specification for [any other files]


## Module contents

Available modules in this package.

Modules are designed to be standard ways of isolating self-contained
pieces of code. This is different from the libraries present within
the `lib` folder. These are supposed to be used by all the other
modules.

Modules are executed automatically when the main program is executed,
based upon the specification of the file `config/modules.json`. This
file comprises of a list of JSON objects, each specifying a module.
An example of the JSON block is:

```
"moduleName" : "module1",
"path"       : "modules/module1/module1.py",
"execute"    : true ,
"description": "",
"owner"      : ""
```

The `path` refers to the location where the module is located. This will
be dynamically loaded if the `execute` parameter is `true`. Each module
function has to have a function called `main()` which will be executed
once the module is loaded. Whatever needs to be done for running the
module should be done from within this main function.

When distributiong a system, it is typically a good idea to make sure that
all the modules have `execute` set to `false`. This will prevent modules
from being accidently executed.

### Available Modules:

The following modules are available. Please check the respective
modules for detailed description of how to operate the modules.

#### module1

This is a test module that comes with the cookiecutter to be used as
a template for other modules.
