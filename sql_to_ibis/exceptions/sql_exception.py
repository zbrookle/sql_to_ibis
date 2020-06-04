"""
Exceptions for SQL to Pandas
"""
from typing import List


class MultipleQueriesException(Exception):
    """
    Raised when multiple queries are passed into sql to pandas.
    """

    def __init__(self):
        Exception.__init__(self, "Only one sql statement may be entered")


class InvalidQueryException(Exception):
    """
    Raised when an invalid query is passed into a sql to pandas.
    """

    def __init__(self, message):
        Exception.__init__(self, f"Invalid query!\n{message}")


class TableExprDoesNotExist(Exception):
    """
    Raised when a DataFrame doesn't exist
    """

    def __init__(self, table_name):
        Exception.__init__(self, f"Table {table_name} has not been defined")

class AmbiguousColumnException(Exception):
    """
    Raised when a column name is not specific enough
    """
    def __init__(self, columnn: str, possible_tables: List[str]):
        Exception.__init__(self, f"For column {columnn}, one of {possible_tables} "
                                 f"must be specified")