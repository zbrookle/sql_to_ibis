"""
Module containing all sql objects
"""
from typing import Any, List, Set

import ibis
from ibis.expr.types import AnyColumn, NumericScalar
from ibis.expr.window import Window as IbisWindow
from sql_to_ibis.sql.sql_clause_objects import (
    OrderByExpression,
    ColumnExpression,
    PartitionByExpression,
)
from sql_to_ibis.sql.sql_value_objects import Table


class AliasRegistry:
    def __init__(self):
        self._registry = {}

    def add_to_registry(self, alias: str, table: Table):
        assert alias not in self._registry
        self._registry[alias] = table

    def get_registry_entry(self, item: str):
        return self._registry[item]

    def __contains__(self, item):
        return item in self._registry

    def __repr__(self):
        return f"Registry:\n{self._registry}"


class AmbiguousColumn:
    """
    Class for identifying ambiguous table names
    """

    def __init__(self, tables: Set[str]) -> None:
        assert tables != set()
        self._tables = tables

    def __repr__(self) -> str:
        return f"AmbiguousColumn({', '.join(self.tables)})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AmbiguousColumn) and self.tables == other.tables

    def add_table(self, table):
        self._tables.add(table)

    def remove_table(self, table: str):
        if len(self._tables) <= 1:
            raise Exception("Ambiguous column table set cannot be empty!")
        self._tables.remove(table)

    @property
    def tables(self):
        return self._tables


class Window:
    def __init__(
        self, window_part_list: List[ColumnExpression], aggregation: NumericScalar
    ):
        self.partition: List[AnyColumn] = [
            clause.column_value
            for clause in window_part_list
            if isinstance(clause, PartitionByExpression)
        ]
        self.order_by: List[AnyColumn] = [
            clause.column_value
            for clause in window_part_list
            if isinstance(clause, OrderByExpression)
        ]
        self.aggregation = aggregation

    def apply_ibis_window_function(self) -> IbisWindow:
        return self.aggregation.over(
            ibis.cumulative_window(group_by=self.partition, order_by=self.order_by)
        )
