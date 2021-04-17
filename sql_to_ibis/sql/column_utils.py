from typing import List, Set

from ibis.expr.types import AnyColumn

# from sql_to_ibis.sql.sql_value_objects import TableOrJoinbase


def rename_duplicates(
    table,  # TODO Come back here and fix the type
    duplicates: Set[str],
    table_name: str,
    table_columns: List[AnyColumn],
):
    for i, column in enumerate(table.column_names):
        if column in duplicates:
            table_columns[i] = table_columns[i].name(f"{table_name}.{column}")
    return table_columns
