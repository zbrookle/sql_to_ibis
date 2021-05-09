sql_to_ibis
===========

.. image:: https://github.com/zbrookle/sql_to_ibis/workflows/CI/badge.svg?branch=master
    :target: https://github.com/zbrookle/sql_to_ibis/actions?query=workflow

.. image:: https://pepy.tech/badge/sql-to-ibis
    :target: https://pepy.tech/project/sql-to-ibis

.. image:: https://img.shields.io/pypi/l/sql_to_ibis.svg
    :target: https://github.com/zbrookle/sql_to_ibis/blob/master/LICENSE.txt

.. image:: https://img.shields.io/pypi/status/sql_to_ibis.svg
    :target: https://pypi.python.org/pypi/sql_to_ibis/

.. image:: https://img.shields.io/pypi/v/sql_to_ibis.svg
    :target: https://pypi.python.org/pypi/sql_to_ibis/

.. image:: https://codecov.io/gh/zbrookle/sql_to_ibis/branch/master/graph/badge.svg?
    :target: https://codecov.io/gh/zbrookle/sql_to_ibis

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

``sql_to_ibis`` is a Python_ package that translates SQL syntax into ibis_ expressions.
This provides the capability of using only one SQL dialect to target many different
backends

.. _Python: https://www.python.org/
.. _ibis: https://github.com/ibis-project/ibis

Installation
------------

.. code-block:: bash

    pip install sql_to_ibis

Usage
-----

Registering and removing temp tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use an ibis table in sql_to_ibis you must register it. Note that for joins or
queries that involve more than one table you must use the same ibis client when
creating both ibis tables. Once the table is registered you can query it using SQL
with the *query* function. In the example below, we create and query a pandas DataFrame

.. code-block:: python

    import ibis.pandas.api
    import pandas
    import sql_to_ibis

    df = pandas.DataFrame({"column1": [1, 2, 3], "column2": ["4", "5", "6"]})
    ibis_table = ibis.pandas.api.from_dataframe(
        df, name="my_table", client=ibis.pandas.api.PandasClient({})
    )
    sql_to_ibis.register_temp_table(ibis_table, "my_table")
    sql_to_ibis.query(
        "select column1, cast(column2 as integer) + 1 as my_col2 from my_table"
    ).execute()

This would output a DataFrame that looks like:

+---------+---------+
| column1 | my_col2 |
+=========+=========+
| 1       | 5       |
+---------+---------+
| 2       | 6       |
+---------+---------+
| 3       | 7       |
+---------+---------+


SQL Syntax
----------
The sql syntax for sql_to_ibis is as follows (Note that all syntax is case insensitive):

Select statement
~~~~~~~~~~~~~~~~

.. code-block:: SQL

    SELECT [{ ALL | DISTINCT }]
        { [ <expression> ] | <expression> [ [ AS ] <alias> ] } [, ...]
    [ FROM <from_item>  [, ...] ]
    [ WHERE <bool_expression> ]
    [ GROUP BY { <expression> [, ...] } ]
    [ HAVING <bool_expression> ]


Example:

.. code-block:: SQL

    SELECT
        column4,
        Sum(column1)
    FROM
        my_table
    WHERE
        column3 = 'yes'
        AND column2 = 'no'
    GROUP BY
        column4

Note that columns with spaces in them can be expressed using double quotes. For example:

.. code-block:: SQL

    SELECT
        "my column 1",
        column2 as "the second column"
    FROM
        my_table


Set operations
~~~~~~~~~~~~~~

.. code-block:: SQL

    <select_statement1>
    {UNION [DISTINCT] | UNION ALL | INTERSECT [DISTINCT] | EXCEPT [DISTINCT] | EXCEPT ALL}
    <select_statment2>


Example:

.. code-block:: SQL

    SELECT
        *
    FROM
        table1
    UNION
    SELECT
        *
    FROM
        table2


Joins
~~~~~

.. code-block:: SQL

    INNER, CROSS, FULL OUTER, LEFT OUTER, RIGHT OUTER, FULL, LEFT, RIGHT


.. code-block:: SQL

    SELECT
       *
    FROM
       table1
       CROSS JOIN
          table2


.. code-block:: SQL

    SELECT
        *
    FROM
        table1
    JOIN
        table2
            ON table1.column1 = table2.column1


Order by and limit
~~~~~~~~~~~~~~~~~~

.. code-block:: SQL

    <set>
    [ORDER BY <expression>]
    [LIMIT <number>]


Example:

.. code-block:: SQL

    SELECT
       *
    FROM
       table1
    ORDER BY
       column1
    LIMIT 5

Windowed aggregation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: SQL

    <aggregate>() OVER(
            [PARTITION BY (<expresssion> [, <expression>...)]
            [ORDER_BY (<expresssion> [, <expression>...)]
            [ ( ROWS | RANGE ) ( <preceding> | BETWEEN <preceding> AND <following> ) ]
           )

    <preceding>: UNBOUNDED PRECEDING | <unsigned_integer> PRECEDING | CURRENT ROW
    <following>: UNBOUNDED FOLLOWING | <unsigned_integer> FOLLOWING | CURRENT ROW


Supported expressions and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: SQL

    +, -, *, /

.. code-block:: SQL

    CASE WHEN <condition> THEN <result> [WHEN ...] ELSE <result> END

.. code-block:: SQL

    SUM, AVG, MIN, MAX

.. code-block:: SQL

    {RANK | DENSE_RANK} OVER([PARTITION BY (<expresssion> [, <expression>...])])

.. code-block:: SQL

    CAST (<expression> AS <data_type>)

.. code-block:: SQL

    <expression> is null

.. code-block:: SQL

    <expression> is not null

.. code-block:: SQL

    COALESCE(<expresssion> [, <expression>...])

* Anything in <> is meant to be some string
* Anything in [] is optional
* Anything in {} is grouped together

Supported Data Types for cast expressions include:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* VARCHAR, STRING
* INT16, SMALLINT
* INT32, INT
* INT64, BIGINT
* FLOAT16
* FLOAT32
* FLOAT, FLOAT64
* BOOL
* DATETIME64, TIMESTAMP
* CATEGORY
* OBJECT