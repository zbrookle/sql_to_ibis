"""
Exceptions for SQL to Pandas
"""
from typing import List

from ibis.expr.types import AnyColumn


class InvalidQueryException(Exception):
    """
    Raised when an invalid query is passed into a sql to pandas.
    """

    def __init__(self, message: str) -> None:
        Exception.__init__(self, f"Invalid query!\n{message}")


class NeedsAggOrGroupQueryException(InvalidQueryException):
    def __init__(self, column: AnyColumn) -> None:
        super().__init__(
            f"For column '{column}' you must either group or provide an aggregation"
        )


class TableExprDoesNotExist(Exception):
    """
    Raised when a DataFrame doesn't exist
    """

    def __init__(self, table_name: str) -> None:
        Exception.__init__(self, f"Table {table_name} has not been defined")


class ColumnNotFoundError(Exception):
    """
    Raised when a column is not present
    """

    def __init__(self, column_name: str, tables: List[str]) -> None:
        super().__init__(f"Column {column_name} not found in table(s) {tables}")


class AmbiguousColumnException(Exception):
    """
    Raised when a column name is not specific enough
    """

    def __init__(self, column: str, possible_tables: List[str]) -> None:
        super().__init__(
            f"For column '{column}', one of {possible_tables} must be specified",
        )


class UnsupportedColumnOperation(Exception):
    def __init__(self, column_type: type, operation: str) -> None:
        super().__init__(
            f"Operation {operation} is not supported for "
            f"columns of type {column_type}"
        )
