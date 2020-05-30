# sql_to_ibis

![CI](https://github.com/zbrookle/sql_to_ibis/workflows/CI/badge.svg)
[![Downloads](https://pepy.tech/badge/dataframe-sql)](https://pepy.tech/project/sql
-to-ibis)
[![PyPI license](https://img.shields.io/pypi/l/sql_to_ibis.svg)](https://pypi.python.org/pypi/sql_to_ibis/)
[![PyPI status](https://img.shields.io/pypi/status/sql_to_ibis.svg)](https://pypi.python.org/pypi/sql_to_ibis/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/sql_to_ibis.svg)](https://pypi.python.org/pypi/sql_to_ibis/)

## Installation

```bash
pip install sql_to_ibis
```

## SQL Syntax
The sql syntax for sql_to_ibis is as follows:

Select statement:

```SQL
SELECT [{ ALL | DISTINCT }]
    { [ <expression> ] | <expression> [ [ AS ] <alias> ] } [, ...]
[ FROM <from_item>  [, ...] ]
[ WHERE <bool_expression> ]
[ GROUP BY { <expression> [, ...] } ]
[ HAVING <bool_expression> ]
```

Set operations:

```SQL
<select_statement1>
{UNION [DISTINCT] | UNION ALL | INTERSECT [DISTINCT] | EXCEPT [DISTINCT] | EXCEPT ALL}
<select_statment2>
```

Joins:

```SQL
INNER, CROSS, FULL OUTER, LEFT OUTER, RIGHT OUTER, FULL, LEFT, RIGHT
```

Order by and limit:

```SQL
<set>
[ORDER BY <expression>]
[LIMIT <number>]
```

Supported expressions and functions:
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

### Issues
* Ibis needs a select columns operator not just a drop one
* Ibis needs a way of choosing column order