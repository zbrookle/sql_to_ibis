from dataclasses import dataclass
from typing import Optional, Union

from sql_to_ibis.sql.sql_value_objects import (
    Column,
    NestedJoinBase,
    Subquery,
    Table,
    Value,
)


@dataclass
class LimitExpression:
    limit: int


@dataclass
class ValueExpression:
    value: Value


@dataclass
class WhereExpression(ValueExpression):
    pass


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


@dataclass
class ExtentExpression:
    extent: Optional[int]


@dataclass
class Following(ExtentExpression):
    extent: Optional[int] = 0


@dataclass
class Preceding(ExtentExpression):
    extent: Optional[int] = None


@dataclass
class FrameExpression:
    frame_type: str = "range"
    preceding: Preceding = Preceding()
    following: Following = Following()


@dataclass
class FromExpression:
    value: Union[Subquery, NestedJoinBase, Table]


@dataclass
class AliasExpression:
    alias: str
