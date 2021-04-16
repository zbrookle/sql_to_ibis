"""
Convert sql query to an ibis expression
"""
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict

from ibis.expr.types import TableExpr
from lark import Lark, UnexpectedToken
from lark.exceptions import VisitError

from sql_to_ibis.exceptions.sql_exception import InvalidQueryException
from sql_to_ibis.parsing.sql_parser import SQLTransformer
from sql_to_ibis.sql.sql_objects import AmbiguousColumn
from sql_to_ibis.sql.sql_value_objects import Table

_ROOT = Path(__file__).parent
GRAMMAR_PATH = os.path.join(_ROOT, "grammar", "sql.lark")
with open(file=GRAMMAR_PATH) as sql_grammar_file:
    _GRAMMAR_TEXT = sql_grammar_file.read()


def register_temp_table(table: TableExpr, table_name: str):
    """
    Registers related metadata from a :class: ~`ibis.expr.types.TableExpr` for use with
    SQL

    Parameters
    ----------
    table : :class: ~`ibis.expr.types.TableExpr`
        :class: ~`ibis.expr.types.TableExpr` object to register
    table_name : str
        String that will be used to represent the :class: ~`ibis.expr.types.TableExpr`
        in SQL

    See Also
    --------
    remove_temp_table : Removes all registered metadata related to a table name
    query : Query a registered :class: ~`ibis.expr.types.TableExpr` using an SQL
    interface

    Examples
    --------
    >>> df = pd.read_csv("a_csv_file.csv")
    >>> register_temp_table(df, "my_table_name")
    """
    if not isinstance(table, TableExpr):
        raise TypeError(
            f"Cannot register table of type {type(table)}. Table must be "
            f"of type {TableExpr}"
        )
    table_info = TableInfo()
    table_info.register_temporary_table(table, table_name)


def remove_temp_table(table_name: str):
    """
    Removes all registered metadata related to a table name

    Parameters
    ----------
    table_name : str
        Name of the table to be removed

    See Also
    --------
    register_temp_table : Registers related metadata from a
                          :class: ~`ibis.expr.types.TableExpr` for use with SQL
    query : Query a registered :class: ~`ibis.expr.types.TableExpr` using an SQL
    interface

    Examples
    --------
    >>> remove_temp_table("my_table_name")
    """
    table_info = TableInfo()
    table_info.remove_temp_table(table_name)


def query(sql: str) -> TableExpr:
    """
    Query a registered :class: ~`ibis.expr.types.TableExpr` using an SQL interface

    Query a registered :class: ~`ibis.expr.types.TableExpr` using the following
    interface based on the following general syntax:
    SELECT
    col_name | expr [, col_name | expr] ...
    [FROM table_reference [, table_reference | join_expr]]
    [WHERE where_condition]
    [GROUP BY {col_name | expr }, ... ]
    [HAVING where_condition]
    [ORDER BY {col_name | expr | position}
      [ASC | DESC], ... ]
    [LIMIT {[offset,] row_count | row_count OFFSET offset}]
    [ (UNION ( [DISTINCT] | ALL ) | INTERSECT ( [DISTINCT] | ALL ) |
      EXCEPT ( [DISTINCT] | ALL ) ]
    select_expr


    Parameters
    ----------
    sql : str
        SQL string querying the :class: ~`ibis.expr.types.TableExpr`

    Returns
    -------
    :class: ~`ibis.DataFrame`
        The :class: ~`ibis.expr.types.TableExpr` resulting from the SQL query provided


    """
    return SqlToTable(sql).ibis_expr


class SqlToTable:
    parser = Lark(_GRAMMAR_TEXT, parser="lalr")

    def __init__(self, sql: str):
        self.sql = sql

        self.ast = self.parse_sql()
        self.ibis_expr = self.ast

    def parse_sql(self):
        try:
            tree = self.parser.parse(self.sql)

            table_info = TableInfo()

            return SQLTransformer(
                table_info.ibis_table_name_map.copy(),
                table_info.ibis_table_map.copy(),
                table_info.column_name_map.copy(),
                deepcopy(table_info.column_to_table_name)  # Need deep copy so that
                # ambiguous column references are not distorted
            ).transform(tree)
        except UnexpectedToken as err:
            message = (
                f"Expected one of the following input(s): {err.expected}\n"
                f"Unexpected input at line {err.line}, column {err.column}\n"
                f"{err.get_context(self.sql)}"
            )
            raise InvalidQueryException(message)
        except VisitError as err:
            current_err: Exception = err
            while True:
                if isinstance(current_err, VisitError):
                    current_err = current_err.orig_exc
                else:
                    break
            raise current_err


class TableInfo:
    column_to_table_name: Dict[str, Any] = {}
    column_name_map: Dict[str, Dict[str, str]] = {}
    ibis_table_name_map: Dict[str, str] = {}
    ibis_table_map: Dict[str, Table] = {}

    def add_column_to_column_to_table_name_map(self, column, table):
        if self.column_to_table_name.get(column) is None:
            self.column_to_table_name[column] = table
        elif isinstance(self.column_to_table_name[column], AmbiguousColumn):
            self.column_to_table_name[column].add_table(table)
        else:
            original_table = self.column_to_table_name[column]
            self.column_to_table_name[column] = AmbiguousColumn({original_table, table})

    def register_temporary_table(self, ibis_table, table_name: str):
        if table_name.lower() in self.ibis_table_name_map:
            raise Exception(
                f"A table {table_name.lower()} has already been registered. Keep in "
                f"mind that table names are case insensitive"
            )

        self.ibis_table_name_map[table_name.lower()] = table_name
        self.ibis_table_map[table_name] = Table(value=ibis_table, name=table_name)
        self.column_name_map[table_name] = {}
        for column in ibis_table.columns:
            lower_column = column.lower()
            self.column_name_map[table_name][lower_column] = column
            self.add_column_to_column_to_table_name_map(lower_column, table_name)

    def remove_temp_table(self, table_name: str):
        if table_name.lower() not in self.ibis_table_name_map:
            raise Exception(f"Table {table_name.lower()} is not registered")
        real_table_name = self.ibis_table_name_map[table_name.lower()]

        columns = self.ibis_table_map[real_table_name].get_table_expr().columns
        for column in columns:
            lower_column = column.lower()
            value = self.column_to_table_name[lower_column]
            if isinstance(value, AmbiguousColumn):
                value.remove_table(real_table_name)
                if len(value.tables) == 1:
                    last_remaining_table = list(value.tables)[0]
                    self.column_to_table_name[lower_column] = last_remaining_table
            else:
                del self.column_to_table_name[lower_column]

        del self.ibis_table_name_map[table_name.lower()]
        del self.ibis_table_map[real_table_name]
        del self.column_name_map[real_table_name]
