from typing import Union, Optional

from sql_to_ibis.sql.sql_value_objects import Column
from dataclasses import dataclass


class RowRangeClause:
    _row = "ROW"
    _range = "RANGE"
    _clause_types = [_row, _range]

    def __init__(
        self,
        clause_type: str,
        preceding: Optional[Union[int, tuple]],
        following: Optional[Union[int, tuple]],
    ):
        if clause_type not in self._clause_types:
            raise Exception(f"Type must be one of {self._clause_types}")
        self.preceding = preceding
        self.following = following


@dataclass
class ColumnExpression:
    column: Column

    @property
    def column_value(self):
        return self.column.get_value()


@dataclass
class OrderByExpression(ColumnExpression):
    ascending: bool = True

    @property
    def column_value(self):
        column = self.column if self.ascending else self.column.desc()
        return column.get_value()


class PartitionByExpression(ColumnExpression):
    pass
