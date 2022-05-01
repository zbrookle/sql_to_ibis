from __future__ import annotations

from typing import TYPE_CHECKING, List, Set

from ibis.expr.types import AnyColumn

if TYPE_CHECKING:
    from sql_to_ibis.sql.sql_value_objects import Table


def rename_duplicates(
    table: Table,
    duplicates: Set[str],
    table_name: str,
    table_columns: List[AnyColumn],
) -> List[AnyColumn]:
    for i, column in enumerate(table.column_names):
        if column in duplicates:
            table_columns[i] = table_columns[i].name(f"{table_name}.{column}")
    return table_columns
