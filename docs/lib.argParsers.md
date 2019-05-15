# lib.argParsers package

## Submodules

## lib.argParsers.addAllParsers module


#### lib.argParsers.addAllParsers.decodeParsers(logger, args)
convert the parser namespace into a dict

This takes the parsed arguments and converts the values
into a dictionary that can be used …


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **args** (*{a parsed object}*) – A Namespace that contains the values of the parsed
      arguments according to the values provided.



* **Returns**

    A doctionary containing a list of all the parsers
    converted into their respective sub dictionaries



* **Return type**

    dict



#### lib.argParsers.addAllParsers.parsersAdd(logger, parser)
add all available CLI arguments to the parser

This function is going to add all available parser
information into the provided parser argument.


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **parser** (*{argparse.ArgumentParser instance}*) – An instance of `argparse.ArgumentParser()` that will be
      used for parsing the command line arguments.



* **Returns**

    This is a `argparse.ArgumentParser()` instance that captures
    all the optional argument options that have been passed to
    the instance



* **Return type**

    `argparse.ArgumentParser()` instance



#### lib.argParsers.addAllParsers.updateArgs(logger, defaultDict, claDict)
helper function for decoding arguments

This function takes the dictionary provided by the
namespace arguments, and updates the dictionary that
needs parsing, in a meaningful manner. This allows
`str`, `bool`, `int`, `float`, `complex`
and `dict` arguments to be changed. Make sure that
you use it with caution. If you are unsure what this
is going to return, just role your own parser.


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **defaultDict** (*{**[**type**]**}*) – [description]

    * **claDict** (*{**[**type**]**}*) – [description]



* **Returns**

    [description]



* **Return type**

    [type]


## lib.argParsers.config module


#### lib.argParsers.config.addParsers(logger, parser)
add argument parsers specific to the `config/config.json` file

This function is kgoing to add argument parsers specific to the
`config/config.json` file. This file has several options for
logging data. This information will be


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **parser** (*{argparse.ArgumentParser instance}*) – An instance of `argparse.ArgumentParser()` that will be
      used for parsing the command line arguments specific to the
      config file



* **Returns**

    The same parser argument to which new CLI arguments have been
    appended



* **Return type**

    argparse.ArgumentParser instance



#### lib.argParsers.config.decodeParser(logger, args)
generate a dictionary from the parsed args

The parsed args may/may not be present. When they are
present, they are pretty hard to use. For this reason,
this function is going to convert the result into
something meaningful.


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **args** (*{args Namespace}*) – parsed arguments from the command line



* **Returns**

    Dictionary that converts the arguments into something
    meaningful



* **Return type**

    dict


## lib.argParsers.dqnAgent module


#### lib.argParsers.dqnAgent.addParsers(logger, parser)
add argument parsers specific to the `config/config.json` file

This function is kgoing to add argument parsers specific to the
`config/config.json` file. This file has several options for
logging data. This information will be


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **parser** (*{argparse.ArgumentParser instance}*) – An instance of `argparse.ArgumentParser()` that will be
      used for parsing the command line arguments specific to the
      config file



* **Returns**

    The same parser argument to which new CLI arguments have been
    appended



* **Return type**

    argparse.ArgumentParser instance



#### lib.argParsers.dqnAgent.decodeParser(logger, args)
generate a dictionary from the parsed args

The parsed args may/may not be present. When they are
present, they are pretty hard to use. For this reason,
this function is going to convert the result into
something meaningful.


* **Parameters**

    * **logger** (*{logging.Logger}*) – The logger used for logging error information

    * **args** (*{args Namespace}*) – parsed arguments from the command line



* **Returns**

    Dictionary that converts the arguments into something
    meaningful



* **Return type**

    dict


## Module contents

Argument parsers will be located gere

Currently, there is just one argument parser. However
you are encouraged to use as many argument parsers
that you wish to have. Ideally you should have one
argument parser per config file.

For each config file, you want to add a function for adding
the respective parser to the CLI and another that will
convert the values back into a dictionary. You have to
generate a proper namespace for your parsed documents
so that they can be easily separated. We recommend using
the name of the config file without the extension as a
starting point.

### Defining Parsers

Parsers for a new config file should be defined within a new
file correcponding to that file. For example, a `config.json`
file comes with a file `config.py` Each file should contain
two functions:

> * `addParsers(parser)`

> * `decodeParser(args)`

The `addParser()` function will add all the necessary command
line arguments to the supplied `argparse.ArgumentParser` object
and return the object.

The `decodeParser()` function will take a parsed Namespace
object and convert it inot a dictionary.

This way, different parsing arguments can easily be added and
deleted at will without restricting the workflow to a great
extent.

Within the function `addAllParsers.parsersAdd()`  insert all
the individual parser insertion function that you just created.
Within the `addAllParsers.decodeParsers` function, update the
dictionary that it returns containing all the parsed arguments
within one big dictionary. Note that this is going to add values
within the main dictionary only if a particular command line
argument is supplied.

### Defining CLI Options

There should be proper namespace created for the CLI arguments
or else there is a high possibility that the CLI named arguments
are going to collide. For overcoming this, a couple of simple
rules should be followed:

1. Make sure that a CLI argument is always verbose. (Dont use
   one letter abbreviations unless it is a really common option
   and is universally used: like `-v` for `--verbose`, `-h`
   for `--help` etc. Note that in this vase, `-v` is already
   handled by the logging level).

1. Start with the name of the config file. Each config file is
   typically a `json` object that translates into a Python
   `dict`. Hence, start every CLI argument with the name of
   the config file followed by an underscore.

   **For example**: CLI arguments corresponding to `config.json`
   should start with `--config_`

1. Each subsequent object within the main object should have
   the same nomenclature. This will allow a one-to-one mapping
   of a variable in the config file to a CLI argument.

   **For example**: within `config.json`, the information referred
   to by the `logFolder` object within the JSON structure
   `{"logging":{"specs":{"file":{"logFolder":"logs"}}}}` should be
   named `--config_logging_specs_file_logFolder`
