# lib.resultGraph package

## Submodules

## lib.resultGraph.graphLib module


#### lib.resultGraph.graphLib.generateGraph(logger)
generate a directed graph from the modules config

generate a networkX.Graph object by reading the contents
of the `config/modules/` folder.


* **Parameters**

    **logger** (*{logging.logger}*) – logging element



* **Returns**

    Graph of which object is created when



* **Return type**

    networkX.Graph object



#### lib.resultGraph.graphLib.generateSubGraph(logger, graph, keyNode)
generate a subgraph that contains all prior nodes


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **graph** (*{networkX.Graph object}*) – [description]

    * **keyNode** (*{str}*) – Name of the node whose ancestors need to be geenrated.



* **Returns**

    graph containing the particular node and its ancistors.



* **Return type**

    networkX.Graph object



#### lib.resultGraph.graphLib.graphToSerialized(logger, graph)
serializes a graph

Takes a networkX.Graph object and converts it into a serialized
set of nodes and edges.


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **graph** (*{networkX.Graph object}*) – A networkX graph object that is to be serialized



* **Returns**

    This is a tuple of nodes and edges in a serialized format
    that can be later directly inserted into a database.



* **Return type**

    tuple of serialized lists



#### lib.resultGraph.graphLib.plotGraph(logger, graph, fileName=None)
plot the graph


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **graph** (*{networkX.Graph object}*) – The graph that needs to be plotted

    * **fileName** (*{str}**, **optional*) – name of the file where to save the graph (the default is None, which
      results in no graph being generated)



#### lib.resultGraph.graphLib.serializedToGraph(logger, nodes, edges)
deserialize a graph serialized earlier

Take serialized versions of the nodes and edges which is
produced by the function `graphToSerialized` and convert
that into a normal `networkX`.


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **nodes** (*{list}*) – serialized versions of the nodes of a graph

    * **edges** (*{list}*) – A list of edges in a serialized form



* **Returns**

    Takes a list of serialized nodes and edges and converts it
    into a networkX.Graph object



* **Return type**

    networkX.Graph object



#### lib.resultGraph.graphLib.uploadGraph(logger, graph, dbName=None)
upload the supplied graph to a database

Given a graph, this function is going to upload the graph into
a particular database. In case a database is not specified, this
will try to upload the data into the default database.


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **graph** (*{networkX graph}*) – The graph that needs to be uploaded into the database

    * **dbName** (*{str}**, **optional*) – The name of the database in which to upload the data into. This
      is the identifier within the `db.json` configuration file. (the
      default is `None`, which would use the default database specified
      within the same file)


## Module contents

Utilities for generating graphs

This provides a set of utilities that will allow us to geenrate a
girected graph. This assumes that configuration files for all the
modules are present in the `config/modules/` folder. The files
should be JSON files with the folliwing specifications:

```
{
    "inputs"  : {},
    "outputs" : {},
    "params"  : {}
}
```

The `inputs` and the `outputs` refer to the requirements of the
module and the result of the module. Both can be empty, but in that
case, they should be represented by empty dictionaries as shown above.

All the configuration paramethers for a particular module should go
into the dictionary that is referred to by the keyword `params`.

An examples of what can possibly go into the `inputs` and `outputs`
is as follows:

```
"inputs": {
    "abc1":{
        "type"        : "file-csv",
        "location"    : "../data/abc1.csv",
        "description" : "describe how the data is arranged"
    }
},
"outputs" : {
    "abc2":{
        "type"        : "dbTable",
        "location"    : "<dbName.schemaName.tableName>"
        "dbHost"      : "<dbHost>",
        "dbPort"      : "<dbHost>",
        "dbName"      : "<dbName>",
        "description" : "description of the table"
    },
    "abc3":{
        "type"        : "file-png",
        "location"    : "../reports/img/Fig1.png",
        "description" : "some description of the data"
    }
},
"params" : {}
```

In the above code block, the module will comprise of a single input with
the name `abc1` and outputs with names `abc2` and `abc3`. Each of
these objects are associated with two mandatory fields: `type` and
`location`. Each `type` will typically have a meaningful `location`
argument associated with it.

### Example `type\`\`s and their corresponding \`\`location` argument:

> * “file-file”         : “<string containing the location>”,

> * “file-fig”          : “<string containing the location>”,

> * “file-csv”          : “<string containing the location>”,

> * “file-hdf5”         : “<string containing the location>”,

> * “file-meta”         : “<string containing the location>”,

> * “folder-checkPoint” : “<string containing the folder>”,

> * “DB-dbTable”        : “<dbName.schemaName.tableName>”,

> * “DB-dbColumn”       : “<dbName.schemaName.tableName.columnName>”

You are welcome to generate new `types\`\`s. Note that anything starting with a \`\`file-`
represents a file within your folder structure. Anything starting with `folder-`
represents a folder. Examples of these include checkpoints of Tensorflow models during
training, etc. Anything starting with a `DB-` represents a traditional database like
Postgres.

It is particularly important to name the different inputs and outputs consistently
throughout, and this is going to help link the different parts of the graph together.

There are functions that allow graphs to be written to the database, and subsequently
retreived. It would then be possible to generate graphs from the entire set of modules.
Dependencies can then be tracked across different progeams, not just across different
modules.

### Uploading graphs to databases:

It is absolutely possible that you would like to upload the graphs into dataabses. This
can be done if the current database that you are working with has the following tables:

```
create schema if not exists graphs;

create table graphs.nodes (
    program_name     text,
    now              timestamp with time zone,
    node_name        text,
    node_type        text,
    summary          text
);

create table graphs.edges (
    program_name     text,
    now              timestamp with time zone,
    node_from        text,
    node_to          text
);
```

There are functions provided that will be able to take the entire graph and upload them
directly into the databases.

#### Available Graph Libraries:

> * `graphLib`: General purpose libraries for constructing graphs from the module

>       configurations.
