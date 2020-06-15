# sql_to_ibis

![CI](https://github.com/zbrookle/sql_to_ibis/workflows/CI/badge.svg)
[![Downloads](https://pepy.tech/badge/sql-to-ibis)](https://pepy.tech/project/sql-to-ibis)
[![PyPI license](https://img.shields.io/pypi/l/sql_to_ibis.svg)](https://pypi.python.org/pypi/sql_to_ibis/)
[![PyPI status](https://img.shields.io/pypi/status/sql_to_ibis.svg)](https://pypi.python.org/pypi/sql_to_ibis/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/sql_to_ibis.svg)](https://pypi.python.org/pypi/sql_to_ibis/)

## Installation

```bash
pip install sql_to_ibis
```

## Usage

### Registering and removing temp tables

To use an ibis table in sql_to_ibis you must register it. Note that for joins or
 queries that involve more than one table you must use the same ibis client when
  creating both ibis tables. Once the table is registered you can query it using SQL
   with the *query* function. In the example below, we create and query a pandas
    DataFrame

```python
from ibis.pandas.api import from_dataframe, PandasClient
from pandas import read_csv
from sql_to_ibis import register_temp_table, query

df = read_csv("some_file.csv")
ibis_table = from_dataframe(df, name="my_table", client=PandasClient({}))
register_temp_table(ibis_table, "my_table")
query("select column1, column2 as my_col2 from my_table")
```

## SQL Syntax
The sql syntax for sql_to_ibis is as follows (Note that all syntax is case insensitive):

#### Select statement:

```SQL
SELECT [{ ALL | DISTINCT }]
    { [ <expression> ] | <expression> [ [ AS ] <alias> ] } [, ...]
[ FROM <from_item>  [, ...] ]
[ WHERE <bool_expression> ]
[ GROUP BY { <expression> [, ...] } ]
[ HAVING <bool_expression> ]
```

Example:
```SQL
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
```

#### Set operations:

```SQL
<select_statement1>
{UNION [DISTINCT] | UNION ALL | INTERSECT [DISTINCT] | EXCEPT [DISTINCT] | EXCEPT ALL}
<select_statment2>
```

Example
```SQL
SELECT
    *
FROM
    table1
UNION
SELECT
    *
FROM
    table2
```

#### Joins:

```SQL
INNER, CROSS, FULL OUTER, LEFT OUTER, RIGHT OUTER, FULL, LEFT, RIGHT
```

Example:

```SQL
SELECT
   *
FROM
   table1
   CROSS JOIN
      table2
```

```SQL
SELECT
    *
FROM
    table1
JOIN
    table2
        ON table1.column1 = table2.column1
```

#### Order by and limit:

```SQL
<set>
[ORDER BY <expression>]
[LIMIT <number>]
```

Example:

```SQL
SELECT
   *
FROM
   table1
ORDER BY
   column1
LIMIT 5
```

#### Supported expressions and functions:
```SQL
+, -, *, /
```
```SQL
CASE WHEN <condition> THEN <result> [WHEN ...] ELSE <result> END
```
```SQL
SUM, AVG, MIN, MAX
```
```SQL
{RANK | DENSE_RANK} OVER([PARTITION BY (<expresssion> [, <expression>...)])
```
```SQL
CAST (<expression> AS <data_type>)
```
*Anything in <> is meant to be some string <br>
*Anything in [] is optional <br>
*Anything in {} is grouped together

### Supported Data Types for cast expressions include:
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