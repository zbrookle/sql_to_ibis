from typing import List

import ibis
from ibis.expr.types import AnyColumn, TableExpr

from sql_to_ibis import query
from sql_to_ibis.tests.utils import (
    assert_ibis_equal_show_diff,
    assert_state_not_change,
    get_columns_with_alias,
)


@assert_state_not_change
def test_select_star(forest_fires: TableExpr) -> None:
    """
    Tests the simple select * case
    :return:
    """
    my_table = query("select * from forest_fires")
    ibis_table = forest_fires
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_case_insensitivity(forest_fires: TableExpr) -> None:
    """
    Tests to ensure that the sql is case insensitive for table names
    :return:
    """
    my_table = query("select * from FOREST_fires")
    ibis_table = forest_fires
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_specific_fields(forest_fires: TableExpr) -> None:
    """
    Tests selecting specific fields
    :return:
    """
    my_table = query("select temp, RH, wind, rain as water, area from forest_fires")

    ibis_table = forest_fires[["temp", "RH", "wind", "rain", "area"]].relabel(
        {"rain": "water"}
    )
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_star_from_multiple_tables(
    digimon_move_mon_join_columns: List[AnyColumn],
    digimon_move_list: TableExpr,
    digimon_mon_list: TableExpr,
) -> None:
    """
    Test selecting from two different tables
    :return:
    """
    my_table = query("""select * from digimon_mon_list, digimon_move_list""")
    ibis_table = ibis.cross_join(digimon_mon_list, digimon_move_list)[
        digimon_move_mon_join_columns
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_select_columns_from_two_tables_with_same_column_name(
    forest_fires: TableExpr,
) -> None:
    """
    Test selecting tables
    :return:
    """
    my_table = query("""select * from forest_fires table1, forest_fires table2""")
    ibis_table = forest_fires.cross_join(forest_fires)[
        get_columns_with_alias(forest_fires, "table1")
        + get_columns_with_alias(forest_fires, "table2")
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_select_columns_from_three_with_same_column_name(
    forest_fires: TableExpr,
) -> None:
    """
    Test selecting tables
    :return:
    """
    my_table = query(
        """select * from forest_fires table1, forest_fires table2, forest_fires
        table3"""
    )
    ibis_table = forest_fires.cross_join(forest_fires).cross_join(forest_fires)[
        get_columns_with_alias(forest_fires, "table1")
        + get_columns_with_alias(forest_fires, "table2")
        + get_columns_with_alias(forest_fires, "table3")
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_maintain_case_in_query(forest_fires: TableExpr) -> None:
    """
    Test nested subqueries
    :return:
    """
    my_table = query("""select wind, rh from forest_fires""")
    ibis_table = forest_fires[["wind", "RH"]].relabel({"RH": "rh"})
    assert_ibis_equal_show_diff(ibis_table, my_table)


# TODO In ibis can't name same column different things in projection
@assert_state_not_change
def test_multiple_aliases_same_column(forest_fires: TableExpr) -> None:
    """
    Test multiple aliases on the same column
    :return:
    """
    my_table = query(
        """
        select wind as my_wind, wind as also_the_wind, wind as yes_wind
        from
        forest_fires
        """
    )
    wind_column = forest_fires.wind
    ibis_table = forest_fires.projection(
        [
            wind_column.name("my_wind"),
            wind_column.name("also_the_wind"),
            wind_column.name("yes_wind"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_select_column_with_table(forest_fires: TableExpr) -> None:
    my_table = query("select forest_fires.wind from forest_fires")
    ibis_table = forest_fires[forest_fires.wind]
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_column_with_alias_prefix(forest_fires: TableExpr) -> None:
    my_table = query("select table1.wind from forest_fires table1")
    ibis_table = forest_fires[forest_fires.wind]
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_ambiguous_column_in_database_context(
    digimon_mon_list: TableExpr,
) -> None:
    my_table = query("select attribute from digimon_mon_list")
    ibis_table = digimon_mon_list[digimon_mon_list.Attribute.name("attribute")]
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_star_with_table_specified(time_data: TableExpr) -> None:
    my_table = query("select time_data.* from time_data")
    ibis_table = time_data
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_quoted_column_names(digimon_mon_list: TableExpr) -> None:
    my_table = query('select "Equip Slots", "Lv50 Atk" from digimon_mon_list')
    ibis_table = digimon_mon_list[["Equip Slots", "Lv50 Atk"]]
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_rename_to_string_with_spaces(digimon_mon_list: TableExpr) -> None:
    my_table = query('select digimon as "Digimon Name" from digimon_mon_list')
    ibis_table = digimon_mon_list[[digimon_mon_list.Digimon.name("Digimon Name")]]
    assert_ibis_equal_show_diff(my_table, ibis_table)
