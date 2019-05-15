# lib.databaseIO package

## Submodules

## lib.databaseIO.pgIO module


#### lib.databaseIO.pgIO.commitData(logger, query, values=None, dbName=None)
query data from the database

Query the data over here. If there is a problem with
the data, it is going to return the value of None, and
log the error. Your program needs to check whether
there was an error with the query by checking for a None
return value


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **query** (*{str}*) – The query to be made to the databse

    * **values** (*{tuple** or **list-like}**, **optional*) – Additional values to be passed to the query (the default
      is None)

    * **dbName** (*{str** or **None}**, **optional*) – The name of the database to use. If this is None, the function will
      attempt to read the name from the `defaultDB` item within the
      file `../config/db.json`.



* **Returns**

    A list of tuples containing the values is returned. In case
    there is an error, the error will be logged, and a None will
    be return



* **Return type**

    list or None



#### lib.databaseIO.pgIO.commitDataList(logger, query, values, dbName=None)
query data from the database

Query the data over here. If there is a problem with
the data, it is going to return the value of None, and
log the error. Your program needs to check whether
there was an error with the query by checking for a None
return value


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **query** (*{str}*) – The query to be made to the databse

    * **values** (*{tuple** or **list-like}**, **optional*) – Additional values to be passed to the query (the default
      is None)

    * **dbName** (*{str** or **None}**, **optional*) – The name of the database to use. If this is None, the function will
      attempt to read the name from the `defaultDB` item within the
      file `../config/db.json`.



* **Returns**

    A list of tuples containing the values is returned. In case
    there is an error, the error will be logged, and a None will
    be return



* **Return type**

    list or None



#### lib.databaseIO.pgIO.getAllData(logger, query, values=None, dbName=None)
query data from the database

Query the data over here. If there is a problem with the data, it is going
to return the value of None, and log the error. Your program needs to check
whether  there was an error with the query by checking for a None return
value. Note that the location of the dataabses are assumed to be present
within the file `../config/db.json`.


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **query** (*{str}*) – The query to be made to the databse

    * **values** (*{tuple** or **list-like}**, **optional*) – Additional values to be passed to the query (the default is None)

    * **dbName** (*{str** or **None}**, **optional*) – The name of the database to use. If this is None, the function will
      attempt to read the name from the `defaultDB` item within the
      file `../config/db.json`.



* **Returns**

    A list of tuples containing the values is returned. In case
    there is an error, the error will be logged, and a None will
    be return



* **Return type**

    list or None



#### lib.databaseIO.pgIO.getDataIterator(logger, query, values=None, chunks=100, dbName=None)
Create an iterator from a largish query

This is a generator that returns values in chunks of chunksize `chunks`.


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **query** (*{str}*) – The query to be made to the databse

    * **values** (*{tuple** or **list-like}**, **optional*) – Additional values to be passed to the query (the default
      is None)

    * **chunks** (*{number}**, **optional*) – This is the number of rows that the data is going to return at every call
      if __next__() to this function. (the default is 100)

    * **dbName** (*{str** or **None}**, **optional*) – The name of the database to use. If this is None, the function will
      attempt to read the name from the `defaultDB` item within the
      file `../config/db.json`.



* **Yields**

    *list of tuples* – A list of tuples from the query, with a maximum of `chunks` tuples returned
    at one time.



#### lib.databaseIO.pgIO.getSingleDataIterator(logger, query, values=None, dbName=None)
Create an iterator from a largish query

This is a generator that returns values in chunks of chunksize 1.


* **Parameters**

    * **logger** (*{logging.logger}*) – logging element

    * **query** (*{str}*) – The query to be made to the databse

    * **values** (*{tuple** or **list-like}**, **optional*) – Additional values to be passed to the query (the default
      is None)

    * **dbName** (*{str** or **None}**, **optional*) – The name of the database to use. If this is None, the function will
      attempt to read the name from the `defaultDB` item within the
      file `../config/db.json`.



* **Yields**

    *list of tuples* – A list of tuples from the query, with a maximum of `chunks` tuples returned
    at one time.


## Module contents

Functions for accessing databases

This library contains functions that will allow you to write
high performance code for accessing various databases. Currently
it only has access to Postgres libraries.

Specifications of the locations of the databases are assumed to be present
within the `../config/db.json` file. A `../config/db.template.json` file
has been provided for templating your `db.json` file with this file.

### Available Database Libraries:

> * Postgres: `pgIO`
