"""
Test cases for panda to sql
"""
from datetime import date, datetime

from freezegun import freeze_time
import ibis
from ibis.common.exceptions import IbisTypeError
from ibis.expr.types import TableExpr
import pytest

from sql_to_ibis import query, register_temp_table, remove_temp_table
from sql_to_ibis.sql.sql_objects import AmbiguousColumn
import sql_to_ibis.sql.sql_value_objects
from sql_to_ibis.sql_select_query import TableInfo
from sql_to_ibis.tests.markers import ibis_not_implemented
from sql_to_ibis.tests.utils import (
    assert_ibis_equal_show_diff,
    assert_state_not_change,
    get_all_join_columns_handle_duplicates,
    get_columns_with_alias,
    join_params,
)


def test_add_remove_temp_table(digimon_mon_list):
    """
    Tests registering and removing temp tables
    :return:
    """
    frame_name = "digimon_mon_list"
    real_frame_name = TableInfo.ibis_table_name_map[frame_name]
    remove_temp_table(frame_name)
    tables_present_in_column_to_dataframe = set()
    for column in TableInfo.column_to_table_name:
        table = TableInfo.column_to_table_name[column]
        if isinstance(table, AmbiguousColumn):
            for table_name in table.tables:
                tables_present_in_column_to_dataframe.add(table_name)
        else:
            tables_present_in_column_to_dataframe.add(table)

    # Ensure column metadata is removed correctly
    assert (
        frame_name not in TableInfo.ibis_table_name_map
        and real_frame_name not in TableInfo.ibis_table_map
        and real_frame_name not in TableInfo.column_name_map
        and real_frame_name not in tables_present_in_column_to_dataframe
    )

    registered_frame_name = real_frame_name
    register_temp_table(digimon_mon_list, registered_frame_name)

    assert (
        TableInfo.ibis_table_name_map.get(frame_name.lower()) == registered_frame_name
        and real_frame_name in TableInfo.column_name_map
    )

    assert_ibis_equal_show_diff(
        TableInfo.ibis_table_map[registered_frame_name].get_table_expr(),
        digimon_mon_list,
    )

    # Ensure column metadata is added correctly
    for column in digimon_mon_list.columns:
        assert column == TableInfo.column_name_map[registered_frame_name].get(
            column.lower()
        )
        lower_column = column.lower()
        assert lower_column in TableInfo.column_to_table_name
        table = TableInfo.column_to_table_name.get(lower_column)
        if isinstance(table, AmbiguousColumn):
            assert registered_frame_name in table.tables
        else:
            assert registered_frame_name == table


@assert_state_not_change
def test_select_star(forest_fires):
    """
    Tests the simple select * case
    :return:
    """
    my_table = query("select * from forest_fires")
    ibis_table = forest_fires
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_case_insensitivity(forest_fires):
    """
    Tests to ensure that the sql is case insensitive for table names
    :return:
    """
    my_table = query("select * from FOREST_fires")
    ibis_table = forest_fires
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_specific_fields(forest_fires):
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
def test_type_conversion(forest_fires):
    """
    Tests sql as statements
    :return:
    """
    my_table = query(
        """select cast(temp as int64),
        cast(RH as float64) my_rh,
        wind,
        rain,
        area,
        cast(2.0 as int64) my_int,
        cast(3 as float64) as my_float,
        cast(7 as object) as my_object,
        cast(0 as bool) as my_bool from forest_fires"""
    )
    fire_frame = forest_fires.projection(
        [
            forest_fires.temp.cast("int64").name("temp"),
            forest_fires.RH.cast("float64").name("my_rh"),
            forest_fires.wind,
            forest_fires.rain,
            forest_fires.area,
            ibis.literal(2.0).cast("int64").name("my_int"),
            ibis.literal(3).cast("float64").name("my_float"),
            ibis.literal(7).cast("string").name("my_object"),
            ibis.literal(0).cast("bool").name("my_bool"),
        ]
    )
    assert_ibis_equal_show_diff(my_table, fire_frame)


@assert_state_not_change
def test_using_math(forest_fires):
    """
    Test the mathematical operations and order of operations
    :return:
    """
    my_table = query("select temp, 1 + 2 * 3 as my_number from forest_fires")
    ibis_table = forest_fires[["temp"]]
    ibis_table = ibis_table.mutate(
        (ibis.literal(1) + ibis.literal(2) * ibis.literal(3)).name("my_number")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_distinct(forest_fires):
    """
    Test use of the distinct keyword
    :return:
    """
    my_table = query("select distinct area, rain from forest_fires")
    ibis_table = forest_fires[["area", "rain"]].distinct()
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_columns_maintain_order_chosen(forest_fires):
    my_table = query("select area, rain from forest_fires")
    ibis_table = forest_fires[["area", "rain"]]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_subquery(forest_fires):
    """
    Test ability to perform subqueries
    :return:
    """
    my_table = query("select * from (select area, rain from forest_fires) rain_area")
    ibis_table = forest_fires[["area", "rain"]]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@pytest.fixture
def digimon_move_mon_join_columns(digimon_mon_list, digimon_move_list):
    return get_all_join_columns_handle_duplicates(
        digimon_mon_list, digimon_move_list, "DIGIMON_MON_LIST", "DIGIMON_MOVE_LIST"
    )


@join_params
@assert_state_not_change
def test_joins(
    digimon_move_mon_join_columns,
    sql_join: str,
    ibis_join: str,
    digimon_move_list,
    digimon_mon_list,
):
    my_table = query(
        f"select * from digimon_mon_list {sql_join} join "
        "digimon_move_list on "
        "digimon_mon_list.attribute = digimon_move_list.attribute"
    )
    ibis_table = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_mon_list.Attribute == digimon_move_list.Attribute,
        how=ibis_join,
    )[digimon_move_mon_join_columns]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@join_params
@assert_state_not_change
def test_join_specify_selection(
    sql_join: str, ibis_join: str, digimon_move_list, digimon_mon_list
):
    """
    Test join
    :return:
    """
    my_table = query(
        f"""select power from digimon_mon_list {sql_join} join
            digimon_move_list
            on digimon_mon_list.attribute = digimon_move_list.attribute"""
    )
    ibis_table = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_mon_list.Attribute == digimon_move_list.Attribute,
        how=ibis_join,
    )[digimon_move_list.Power.name("power")]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@join_params
@assert_state_not_change
def test_join_wo_specifying_table(
    digimon_move_mon_join_columns,
    sql_join: str,
    ibis_join: str,
    digimon_move_list,
    digimon_mon_list,
):
    """
    Test join where table isn't specified in join
    :return:
    """
    my_table = query(
        f"""
        select * from digimon_mon_list {sql_join} join
        digimon_move_list
        on mon_attribute = move_attribute
        """
    )
    ibis_table = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_mon_list.mon_attribute == digimon_move_list.move_attribute,
        how=ibis_join,
    )[digimon_move_mon_join_columns]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_cross_joins(
    digimon_move_mon_join_columns, digimon_move_list, digimon_mon_list
):
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_table = query(
        """select * from digimon_mon_list cross join
            digimon_move_list"""
    )
    ibis_table = digimon_mon_list.cross_join(digimon_move_list)[
        digimon_move_mon_join_columns
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_cross_join_with_selection(digimon_move_list, digimon_mon_list):
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_table = query(
        """select power from digimon_mon_list cross join
            digimon_move_list"""
    )
    ibis_table = digimon_mon_list.cross_join(digimon_move_list)[
        digimon_move_list.Power.name("power")
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_group_by(forest_fires):
    """
    Test group by constraint
    :return:
    """
    my_table = query("""select month, day from forest_fires group by month, day""")
    ibis_table = forest_fires[["month", "day"]].distinct()
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_avg(forest_fires):
    """
    Test the avg
    :return:
    """
    my_table = query("select avg(temp) from forest_fires")
    ibis_table = forest_fires.aggregate(
        [forest_fires.get_column("temp").mean().name("_col0")]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_sum(forest_fires):
    """
    Test the sum
    :return:
    """
    my_table = query("select sum(temp) from forest_fires")
    ibis_table = forest_fires.aggregate(
        [forest_fires.get_column("temp").sum().name("_col0")]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_max(forest_fires):
    """
    Test the max
    :return:
    """
    my_table = query("select max(temp) from forest_fires")
    ibis_table = forest_fires.aggregate(
        [forest_fires.get_column("temp").max().name("_col0")]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_min(forest_fires):
    """
    Test the min
    :return:
    """
    my_table = query("select min(temp) from forest_fires")
    ibis_table = forest_fires.aggregate(
        [forest_fires.get_column("temp").min().name("_col0")]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_count(forest_fires):
    """
    Test the min
    :return:
    """
    my_table = query("select count(temp) from forest_fires")
    ibis_table = forest_fires.aggregate(
        [forest_fires.get_column("temp").count().name("_col0")]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_multiple_aggs(forest_fires):
    """
    Test multiple aggregations
    :return:
    """
    my_table = query(
        "select min(temp), max(temp), avg(temp), max(wind) from forest_fires"
    )
    temp_column = forest_fires.get_column("temp")
    ibis_table = forest_fires.aggregate(
        [
            temp_column.min().name("_col0"),
            temp_column.max().name("_col1"),
            temp_column.mean().name("_col2"),
            forest_fires.get_column("wind").max().name("_col3"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_agg_w_groupby_no_select_group_by_column(forest_fires):
    """
    Test using aggregates and group by together
    :return:
    """
    my_table = query(
        "select min(temp), max(temp) from forest_fires group by day, month"
    )
    temp_column = forest_fires.get_column("temp")
    ibis_table = (
        forest_fires.group_by(["day", "month"])
        .aggregate([temp_column.min().name("_col0"), temp_column.max().name("_col1")])
        .drop(["day", "month"])
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_agg_w_groupby_select_group_by_column(forest_fires):
    """
    Test using aggregates and group by together
    :return:
    """
    my_table = query(
        "select min(temp), max(temp), day, month from forest_fires group by day, month"
    )
    temp_column = forest_fires.temp
    ibis_table = forest_fires.group_by(
        [forest_fires.day, forest_fires.month]
    ).aggregate([temp_column.min().name("_col0"), temp_column.max().name("_col1")])
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_agg_w_groupby_select_group_by_column_different_casing(forest_fires):
    """
    Test using aggregates and group by together
    :return:
    """
    my_table = query(
        "select min(temp), max(temp), Day, month from forest_fires group by day, month"
    )
    temp_column = forest_fires.get_column("temp")
    selection_and_grouping = [forest_fires.day.name("Day"), forest_fires.month]
    ibis_table = forest_fires.group_by(selection_and_grouping).aggregate(
        [temp_column.min().name("_col0"), temp_column.max().name("_col1")]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_group_by_casing_with_selection(digimon_move_list, digimon_mon_list):
    my_table = query(
        "select max(power) as power, type from digimon_move_list group by type"
    )
    ibis_table = digimon_move_list.group_by(
        [digimon_move_list.Type.name("type")]
    ).aggregate(digimon_move_list.Power.max().name("power"))
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_agg_group_by_different_casing_in_ibis_schema_group_by(
    digimon_move_list, digimon_mon_list
):
    my_table = query("select max(power) as power from digimon_move_list group by type")
    ibis_table = (
        digimon_move_list.group_by(digimon_move_list.Type.name("type"))
        .aggregate(digimon_move_list.Power.max().name("power"))
        .drop(["type"])
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_where_clause(forest_fires):
    """
    Test where clause
    :return:
    """
    my_table = query("""select * from forest_fires where month = 'mar'""")
    ibis_table = forest_fires[forest_fires.month == "mar"]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_all_boolean_ops_clause(forest_fires):
    """
    Test where clause
    :return:
    """
    my_table = query(
        """select * from forest_fires where month = 'mar' and temp > 8.0 and rain >= 0
        and area != 0 and dc < 100 and ffmc <= 90.1
        """
    )
    ibis_table = forest_fires[
        (forest_fires.month == "mar")
        & (forest_fires.temp > 8.0)
        & (forest_fires.rain >= 0)
        & (forest_fires.area != ibis.literal(0))
        & (forest_fires.DC < 100)
        & (forest_fires.FFMC <= 90.1)
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_order_by(forest_fires):
    """
    Test order by clause
    :return:
    """
    my_table = query(
        """select * from forest_fires order by temp desc, wind asc, area"""
    )
    ibis_table = forest_fires.sort_by([("temp", False), ("wind", True), ("area", True)])
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_limit(forest_fires):
    """
    Test limit clause
    :return:
    """
    my_table = query("""select * from forest_fires limit 10""")
    ibis_table = forest_fires.head(10)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_having_multiple_conditions(forest_fires):
    """
    Test having clause
    :return:
    """
    my_table = query(
        "select min(temp) from forest_fires having min(temp) > 2 and max(dc) < 200"
    )
    having_condition = (forest_fires.temp.min() > 2) & (forest_fires.DC.max() < 200)
    ibis_table = forest_fires.aggregate(
        metrics=forest_fires.temp.min().name("_col0"),
        having=having_condition,
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_having_multiple_conditions_with_or(forest_fires):
    """
    Test having clause
    :return:
    """
    my_table = query(
        "select min(temp) from forest_fires having min(temp) > 2 and "
        "max(dc) < 200 or max(dc) > 1000"
    )
    having_condition = (forest_fires.temp.min() > 2) & (forest_fires.DC.max() < 200) | (
        (forest_fires.DC.max() > 1000)
    )
    ibis_table = forest_fires.aggregate(
        metrics=forest_fires.temp.min().name("_col0"),
        having=having_condition,
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_having_one_condition(forest_fires):
    """
    Test having clause
    :return:
    """
    my_table = query("select min(temp) from forest_fires having min(temp) > 2")
    min_aggregate = forest_fires.temp.min()
    ibis_table = forest_fires.aggregate(
        min_aggregate.name("_col0"), having=(min_aggregate > 2)
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_having_with_group_by(forest_fires):
    """
    Test having clause
    :return:
    """
    my_table = query(
        "select min(temp) from forest_fires group by day having min(temp) > 5"
    )
    ibis_table = (
        forest_fires.groupby("day")
        .having(forest_fires.temp.min() > 5)
        .aggregate(forest_fires.temp.min().name("_col0"))
        .drop(["day"])
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_operations_between_columns_and_numbers(forest_fires):
    """
    Tests operations between columns
    :return:
    """
    my_table = query("""select temp * wind + rain / dmc + 37 from forest_fires""")
    ibis_table = forest_fires.projection(
        (
            forest_fires.temp * forest_fires.wind
            + forest_fires.rain / forest_fires.DMC
            + 37
        ).name("_col0")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_select_star_from_multiple_tables(
    digimon_move_mon_join_columns, digimon_move_list, digimon_mon_list
):
    """
    Test selecting from two different tables
    :return:
    """
    my_table = query("""select * from digimon_mon_list, digimon_move_list""")
    ibis_table = digimon_mon_list.cross_join(digimon_move_list)[
        digimon_move_mon_join_columns
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_select_columns_from_two_tables_with_same_column_name(forest_fires):
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
def test_select_columns_from_three_with_same_column_name(forest_fires):
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
def test_maintain_case_in_query(forest_fires):
    """
    Test nested subqueries
    :return:
    """
    my_table = query("""select wind, rh from forest_fires""")
    ibis_table = forest_fires[["wind", "RH"]].relabel({"RH": "rh"})
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_nested_subquery(forest_fires):
    """
    Test nested subqueries
    :return:
    """
    my_table = query(
        """select * from
            (select wind, rh from
              (select * from forest_fires) fires) wind_rh"""
    )
    ibis_table = forest_fires[["wind", "RH"]].relabel({"RH": "rh"})
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_union(forest_fires):
    """
    Test union in queries
    :return:
    """
    my_table = query(
        """
    select * from forest_fires order by wind desc limit 5
    union
    select * from forest_fires order by wind asc limit 5
    """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(5)
    ibis_table = ibis_table1.union(ibis_table2, distinct=True)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_union_distinct(forest_fires):
    """
    Test union distinct in queries
    :return:
    """
    my_table = query(
        """
        select * from forest_fires order by wind desc limit 5
         union distinct
        select * from forest_fires order by wind asc limit 5
        """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(5)
    ibis_table = ibis_table1.union(ibis_table2, distinct=True)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_union_all(forest_fires):
    """
    Test union distinct in queries
    :return:
    """
    my_table = query(
        """
        select * from forest_fires order by wind desc limit 5
         union all
        select * from forest_fires order by wind asc limit 5
        """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(5)
    ibis_table = ibis_table1.union(ibis_table2)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@ibis_not_implemented
@assert_state_not_change
def test_intersect_distinct(forest_fires):
    """
    Test union distinct in queries
    :return:
    """
    my_table = query(
        """
            select * from forest_fires order by wind desc limit 5
             intersect distinct
            select * from forest_fires order by wind desc limit 3
            """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(5)
    ibis_table = ibis_table1.union(ibis_table2, distinct=True)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@ibis_not_implemented
@assert_state_not_change
def test_except_distinct(forest_fires):
    """
    Test except distinct in queries
    :return:
    """
    my_table = query(
        """
                select * from forest_fires order by wind desc limit 5
                 except distinct
                select * from forest_fires order by wind desc limit 3
                """
    )
    ibis_table1 = (
        forest_fires.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    ibis_table2 = (
        forest_fires.copy().sort_values(by=["wind"], ascending=[False]).head(3)
    )
    ibis_table = (
        ibis_table1[~ibis_table1.isin(ibis_table2).all(axis=1)]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@ibis_not_implemented
@assert_state_not_change
def test_except_all(forest_fires):
    """
    Test except distinct in queries
    :return:
    """
    my_table = query(
        """
                select * from forest_fires order by wind desc limit 5
                 except all
                select * from forest_fires order by wind desc limit 3
                """
    )
    ibis_table1 = (
        forest_fires.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    ibis_table2 = (
        forest_fires.copy().sort_values(by=["wind"], ascending=[False]).head(3)
    )
    ibis_table = ibis_table1[~ibis_table1.isin(ibis_table2).all(axis=1)].reset_index(
        drop=True
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_between_operator(forest_fires):
    """
    Test using between operator
    :return:
    """
    my_table = query(
        """
    select * from forest_fires
    where wind between 5 and 6
    """
    )
    ibis_table = forest_fires.filter(forest_fires.wind.between(5, 6))
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_in_operator(forest_fires):
    """
    Test using in operator in a sql query
    :return:
    """
    my_table = query(
        """
    select * from forest_fires where day in ('fri', 'sun')
    """
    )
    ibis_table = forest_fires[
        forest_fires.day.isin([ibis.literal("fri"), ibis.literal("sun")])
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_in_operator_expression_numerical(forest_fires):
    """
    Test using in operator in a sql query
    :return:
    """
    my_table = query(
        """
    select * from forest_fires where X in (5, 9)
    """
    )
    ibis_table = forest_fires[forest_fires.X.isin((ibis.literal(5), ibis.literal(9)))]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_not_in_operator(forest_fires):
    """
    Test using in operator in a sql query
    :return:
    """
    my_table = query(
        """
    select * from forest_fires where day not in ('fri', 'sun')
    """
    )
    ibis_table = forest_fires[
        forest_fires.day.notin([ibis.literal("fri"), ibis.literal("sun")])
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_case_statement_w_name(forest_fires):
    """
    Test using case statements
    :return:
    """
    my_table = query(
        """
        select case when wind > 5 then 'strong'
        when wind = 5 then 'mid'
        else 'weak' end as wind_strength
        from
        forest_fires
        """
    )
    ibis_table = forest_fires.projection(
        ibis.case()
        .when(forest_fires.wind > 5, "strong")
        .when(forest_fires.wind == 5, "mid")
        .else_("weak")
        .end()
        .name("wind_strength")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_case_statement_w_no_name(forest_fires):
    """
    Test using case statements
    :return:
    """
    my_table = query(
        """
        select case when wind > 5 then 'strong' when wind = 5 then 'mid' else 'weak' end
        from forest_fires
        """
    )
    ibis_table = forest_fires.projection(
        ibis.case()
        .when(forest_fires.wind > 5, "strong")
        .when(forest_fires.wind == 5, "mid")
        .else_("weak")
        .end()
        .name("_col0")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_case_statement_w_other_columns_as_result(forest_fires):
    """
    Test using case statements
    :return:
    """
    my_table = query(
        """
        select case when wind > 5 then month when wind = 5 then 'mid' else day end
        from forest_fires
        """
    )
    ibis_table = forest_fires.projection(
        ibis.case()
        .when(forest_fires.wind > 5, forest_fires.month)
        .when(forest_fires.wind == 5, "mid")
        .else_(forest_fires.day)
        .end()
        .name("_col0")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


# TODO Ibis is showing window as object in string representation
@assert_state_not_change
def test_rank_statement_one_column(forest_fires):
    """
    Test rank statement
    :return:
    """
    my_table = query(
        """
    select wind, rank() over(order by wind) as wind_rank
    from forest_fires
    """
    )
    ibis_table = forest_fires.projection(
        [
            forest_fires.wind,
            forest_fires.wind.rank()
            .over(ibis.window(order_by=[forest_fires.wind]))
            .name("wind_rank"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_rank_statement_many_columns(forest_fires):
    """
    Test rank statement
    :return:
    """
    my_table = query(
        """
    select wind, rain, month, rank() over(order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    ibis_table: TableExpr = forest_fires[["wind", "rain", "month"]]
    ibis_table = ibis_table.projection(
        [
            forest_fires.wind,
            forest_fires.rain,
            forest_fires.month,
            forest_fires.wind.rank()
            .over(
                ibis.window(
                    order_by=[
                        ibis.desc(forest_fires.wind),
                        forest_fires.rain,
                        forest_fires.month,
                    ]
                )
            )
            .name("rank"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_dense_rank_statement_many_columns(forest_fires):
    """
    Test dense_rank statement
    :return:
    """
    my_table = query(
        """
    select wind, rain, month,
    dense_rank() over(order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    ibis_table = forest_fires[["wind", "rain", "month"]]
    ibis_table = ibis_table.projection(
        [
            forest_fires.wind,
            forest_fires.rain,
            forest_fires.month,
            forest_fires.wind.dense_rank()
            .over(
                ibis.window(
                    order_by=[
                        ibis.desc(forest_fires.wind),
                        forest_fires.rain,
                        forest_fires.month,
                    ]
                )
            )
            .name("rank"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_rank_over_partition_by(forest_fires):
    """
    Test rank partition by statement
    :return:
    """
    my_table = query(
        """
    select wind, rain, month, day,
    rank() over(partition by day order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    ibis_table = forest_fires[["wind", "rain", "month", "day"]]
    ibis_table = ibis_table.projection(
        [
            forest_fires.wind,
            forest_fires.rain,
            forest_fires.month,
            forest_fires.day,
            forest_fires.day.rank()
            .over(
                ibis.window(
                    order_by=[
                        ibis.desc(forest_fires.wind),
                        forest_fires.rain,
                        forest_fires.month,
                    ],
                    group_by=[forest_fires.day],
                )
            )
            .name("rank"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_partition_by_multiple_columns(forest_fires):
    """
    Test rank partition by statement
    :return:
    """
    my_table = query(
        """
    select wind, rain, month, day,
    rank() over(partition by day, month order by wind) as rank
    from forest_fires
    """
    )
    ibis_table = forest_fires[["wind", "rain", "month", "day"]]
    ibis_table = ibis_table.projection(
        [
            forest_fires.wind,
            forest_fires.rain,
            forest_fires.month,
            forest_fires.day,
            forest_fires.day.rank()
            .over(
                ibis.window(
                    order_by=[forest_fires.wind],
                    group_by=[forest_fires.day, forest_fires.month],
                )
            )
            .name("rank"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_dense_rank_over_partition_by(forest_fires):
    """
    Test rank partition by statement
    :return:
    """
    my_table = query(
        """
    select wind, rain, month, day,
    dense_rank() over(partition by day order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    ibis_table = forest_fires[["wind", "rain", "month", "day"]]
    ibis_table = ibis_table.projection(
        [
            forest_fires.wind,
            forest_fires.rain,
            forest_fires.month,
            forest_fires.day,
            forest_fires.day.dense_rank()
            .over(
                ibis.window(
                    order_by=[
                        ibis.desc(forest_fires.wind),
                        forest_fires.rain,
                        forest_fires.month,
                    ],
                    group_by=[forest_fires.day],
                )
            )
            .name("rank"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_set_string_value_as_column_value(forest_fires):
    """
    Select a string like 'Yes' as a column value
    :return:
    """
    my_table = query(
        """
    select wind, 'yes' as wind_yes from forest_fires"""
    )
    ibis_table = forest_fires[["wind"]].mutate(ibis.literal("yes").name("wind_yes"))
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_datetime_cast(forest_fires):
    """
    Select casting a string as a date
    :return:
    """
    my_table = query(
        """
    select wind, cast('2019-01-01' as datetime64) as my_date from forest_fires"""
    )
    ibis_table = forest_fires[["wind"]].mutate(
        ibis.literal("2019-01-01").cast("timestamp").name("my_date")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_date_cast(forest_fires):
    """
    Select casting a string as a date
    :return:
    """
    my_table = query(
        """
    select wind, cast('2019-01-01' as date) as my_date from forest_fires"""
    )
    ibis_table = forest_fires[["wind"]].mutate(
        ibis.literal("2019-01-01").cast("date").name("my_date")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_timestamps(forest_fires):
    """
    Select now() as date
    :return:
    """
    with freeze_time(datetime.now()):
        my_table = query(
            """
        select wind, now(), today(), timestamp('2019-01-31', '23:20:32')
        from forest_fires"""
        )
        ibis_table = forest_fires[["wind"]].mutate(
            [
                ibis.literal(datetime.now()).name("now()"),
                ibis.literal(date.today()).name("today()"),
                ibis.literal(datetime(2019, 1, 31, 23, 20, 32)).name("_literal2"),
            ]
        )
        assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_case_statement_with_same_conditions(forest_fires):
    """
    Test using case statements
    :return:
    """
    my_table = query(
        """
        select case when wind > 5 then month when wind > 5 then 'mid' else day end
        from forest_fires
        """
    )
    ibis_table = forest_fires.projection(
        ibis.case()
        .when(forest_fires.wind > 5, forest_fires.month)
        .when(forest_fires.wind > 5, "mid")
        .else_(forest_fires.day)
        .end()
        .name("_col0")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


# TODO In ibis can't name same column different things in projection
@assert_state_not_change
def test_multiple_aliases_same_column(forest_fires):
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


@pytest.mark.xfail(reason="Will be fixed in next ibis release", raises=IbisTypeError)
@assert_state_not_change
def test_sql_data_types(avocado):
    """
    Tests sql data types
    :return:
    """
    my_table = query(
        """
        select
            cast(avocado_id as object) as avocado_id_object,
            cast(avocado_id as int16) as avocado_id_int16,
            cast(avocado_id as smallint) as avocado_id_smallint,
            cast(avocado_id as int32) as avocado_id_int32,
            cast(avocado_id as int) as avocado_id_int,
            cast(avocado_id as int64) as avocado_id_int64,
            cast(avocado_id as bigint) as avocado_id_bigint,
            cast(avocado_id as float) as avocado_id_float,
            cast(avocado_id as float16) as avocado_id_float16,
            cast(avocado_id as float32) as avocado_id_float32,
            cast(avocado_id as float64) as avocado_id_float64,
            cast(avocado_id as bool) as avocado_id_bool,
            cast(avocado_id as category) as avocado_id_category,
            cast(date as date) as date,
            cast(date as datetime64) as datetime,
            cast(date as timestamp) as timestamp,
            cast(date as time) as time,
            cast(region as varchar) as region_varchar,
            cast(region as string) as region_string
        from avocado
        """
    )

    date_column = sql_to_ibis.sql.sql_value_objects.Date
    id_column = avocado.avocado_id
    region_column = avocado.region
    ibis_table = avocado.projection(
        [
            id_column.cast("string").name("avocado_id_object"),
            id_column.cast("int16").name("avocado_id_int16"),
            id_column.cast("int16").name("avocado_id_smallint"),
            id_column.cast("int32").name("avocado_id_int32"),
            id_column.cast("int32").name("avocado_id_int"),
            id_column.cast("int64").name("avocado_id_int64"),
            id_column.cast("int64").name("avocado_id_bigint"),
            id_column.cast("float").name("avocado_id_float"),
            id_column.cast("float16").name("avocado_id_float16"),
            id_column.cast("float32").name("avocado_id_float32"),
            id_column.cast("float64").name("avocado_id_float64"),
            id_column.cast("bool").name("avocado_id_bool"),
            id_column.cast("category").name("avocado_id_category"),
            date_column.cast("date").name("date"),
            date_column.cast("timestamp").name("datetime"),
            date_column.cast("timestamp").name("timestamp"),
            date_column.cast("time").name("time"),
            region_column.cast("string").name("region_varchar"),
            region_column.cast("string").name("region_string"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@pytest.mark.xfail(reason="This needs to be solved", raises=AssertionError)
@pytest.mark.parametrize(
    "sql",
    [
        """
    select * from
    (
    (select X, Y, rain from forest_fires) table1
    join
    (select X, Y, rain from forest_fires) table2
    on table1.x = table2.x) sub
    """,
        """
    select * from
    (select X, Y, rain from forest_fires) table1
    join
    (select X, Y, rain from forest_fires) table2
    on table1.x = table2.x
    """,
    ],
)
@assert_state_not_change
def test_joining_two_subqueries_with_overlapping_columns_same_table(sql, forest_fires):
    my_table = query(sql)
    columns = ["X", "Y", "rain"]

    def get_select_rename_columns(alias: str):
        my_columns = forest_fires.get_columns(columns)
        renamed_columns = []
        for i, column in enumerate(my_columns):
            renamed_columns.append(column.name(f"{alias}.{columns[i]}"))
        return my_columns, renamed_columns

    select1, renamed1 = get_select_rename_columns("table1")
    select2, renamed2 = get_select_rename_columns("table2")
    subquery1 = forest_fires[select1]
    subquery2 = forest_fires[select2]
    joined = subquery1.join(
        subquery2, predicates=subquery1.X == subquery2.X
    ).projection(renamed1 + renamed2)
    assert_ibis_equal_show_diff(joined, my_table)


@pytest.mark.parametrize(
    "sql",
    [
        """
    select * from
    ((select type, attribute, power from digimon_move_list) table1
    join
    (select type, attribute, digimon from
    digimon_mon_list) table2
    on table1.type = table2.type) sub
    """,
        """
    select * from
    ((select type, attribute, power from digimon_move_list) table1
    join
    (select type, attribute, digimon from digimon_mon_list) table2
    on table1.type = table2.type) sub
    """,
    ],
)
@assert_state_not_change
def test_joining_two_subqueries_with_overlapping_columns_different_tables(
    sql, digimon_mon_list, digimon_move_list
):
    my_table = query(sql)
    subquery1 = digimon_move_list[
        [
            digimon_move_list.Type.name("type"),
            digimon_move_list.Attribute.name("attribute"),
            digimon_move_list.Power.name("power"),
        ]
    ]
    subquery2 = digimon_mon_list[
        [
            digimon_mon_list.Type.name("type"),
            digimon_mon_list.Attribute.name("attribute"),
            digimon_mon_list.Digimon.name("digimon"),
        ]
    ]
    ibis_table = subquery1.join(
        subquery2, predicates=subquery1.type == subquery2.type
    ).projection(
        [
            subquery1.type.name("table1.type"),
            subquery1.attribute.name("table1.attribute"),
            subquery1.power.name("power"),
            subquery2.type.name("table2.type"),
            subquery2.attribute.name("table2.attribute"),
            subquery2.digimon,
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_math_order_of_operations_no_parens(avocado):
    """
    Test math parentheses
    :return:
    """

    my_table = query("select 20 * avocado_id + 3 / 20 as my_math from avocado")

    ibis_table = avocado.projection(
        [
            (
                ibis.literal(20) * avocado.avocado_id
                + ibis.literal(3) / ibis.literal(20)
            ).name("my_math")
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_math_order_of_operations_with_parens(avocado):
    """
    Test math parentheses
    :return:
    """

    my_table = query(
        "select 20 * (avocado_id + 3) / (20 + avocado_id) as my_math from avocado"
    )
    avocado_id = avocado.avocado_id
    ibis_table = avocado.projection(
        [
            (
                ibis.literal(20)
                * (avocado_id + ibis.literal(3))
                / (ibis.literal(20) + avocado_id)
            ).name("my_math")
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_boolean_order_of_operations_with_parens(forest_fires):
    """
    Test boolean order of operations with parentheses
    :return:
    """
    my_table = query(
        "select * from forest_fires "
        "where (month = 'oct' and day = 'fri') or "
        "(month = 'nov' and day = 'tue')"
    )

    ibis_table = forest_fires[
        ((forest_fires.month == "oct") & (forest_fires.day == "fri"))
        | ((forest_fires.month == "nov") & (forest_fires.day == "tue"))
    ]

    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_capitalized_agg_functions(digimon_move_list):
    my_table = query("select MAX(type), AVG(power), MiN(power) from DIGImON_move_LiST")
    ibis_table = digimon_move_list.aggregate(
        [
            digimon_move_list.Type.max().name("_col0"),
            digimon_move_list.Power.mean().name("_col1"),
            digimon_move_list.Power.min().name("_col2"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_aggregates_in_subquery(digimon_move_list):
    my_table = query("select * from (select max(power) from digimon_move_list) test")
    ibis_table = digimon_move_list.aggregate(
        digimon_move_list.Power.max().name("_col0")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_column_values_in_subquery(digimon_move_list):
    my_table = query(
        """
    select move, type, power from
    digimon_move_list
    where
    power in
        ( select max(power) as power
         from digimon_move_list
         group by type ) t1
    """
    )
    subquery = (
        digimon_move_list.groupby(digimon_move_list.get_column("Type").name("type"))
        .aggregate(digimon_move_list.Power.max().name("power"))
        .drop(["type"])
    )
    ibis_table = digimon_move_list.filter(
        digimon_move_list.Power.isin(subquery.get_column("power"))
    ).projection(
        [
            digimon_move_list.Move.name("move"),
            digimon_move_list.Type.name("type"),
            digimon_move_list.Power.name("power"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_select_column_with_table(forest_fires):
    my_table = query("select forest_fires.wind from forest_fires")
    ibis_table = forest_fires[forest_fires.wind]
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_column_with_alias_prefix(forest_fires):
    my_table = query("select table1.wind from forest_fires table1")
    ibis_table = forest_fires[forest_fires.wind]
    assert_ibis_equal_show_diff(my_table, ibis_table)


@assert_state_not_change
def test_select_ambiguous_column_in_database_context(digimon_mon_list):
    my_table = query("select attribute from digimon_mon_list")
    ibis_table = digimon_mon_list[digimon_mon_list.Attribute.name("attribute")]
    assert_ibis_equal_show_diff(my_table, ibis_table)


@ibis_not_implemented
@assert_state_not_change
def test_group_by_having(digimon_move_list):
    my_table = query(
        "select type from digimon_move_list group by type having avg(power) > 50"
    )
    ibis_table = (
        digimon_move_list.group_by("Type")
        .aggregate(digimon_move_list.Type.first())
        .having(digimon_move_list.Power.mean() > 50)
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_count_star(forest_fires):
    my_table = query("select count(*) from forest_fires")
    ibis_table = forest_fires.aggregate([forest_fires.count().name("_col0")])
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_count_star_cross_join(digimon_move_list, digimon_mon_list):
    my_table = query(
        "select count(*) from digimon_move_list cross join " "digimon_mon_list"
    )
    cross_join_table = digimon_move_list.cross_join(digimon_mon_list)
    ibis_table = cross_join_table.aggregate([cross_join_table.count().name("_col0")])
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_window_function_partition(time_data):
    my_table = query(
        """SELECT count,
       duration_seconds,
       SUM(duration_seconds) OVER
         (PARTITION BY person) AS running_total,
       COUNT(duration_seconds) OVER
         (PARTITION BY person) AS running_count,
       AVG(duration_seconds) OVER
         (PARTITION BY person) AS running_avg
  FROM time_data"""
    )
    ibis_table = time_data.projection(
        [
            time_data.get_column("count"),
            time_data.duration_seconds,
            time_data.duration_seconds.sum()
            .over(ibis.range_window(group_by=time_data.person, following=0))
            .name("running_total"),
            time_data.duration_seconds.count()
            .over(ibis.range_window(group_by=time_data.person, following=0))
            .name("running_count"),
            time_data.duration_seconds.mean()
            .over(ibis.range_window(group_by=time_data.person, following=0))
            .name("running_avg"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_window_function_partition_order_by(time_data):
    my_table = query(
        """SELECT count,
       duration_seconds,
       SUM(duration_seconds) OVER
         (PARTITION BY person, team ORDER by start_time, end_time) AS running_total,
       COUNT(duration_seconds) OVER
         (PARTITION BY person ORDER by count) AS running_count,
       AVG(duration_seconds) OVER
         (PARTITION BY person ORDER by count) AS running_avg
  FROM time_data"""
    )
    count_column = time_data.get_column("count")
    ibis_table = time_data.projection(
        [
            count_column,
            time_data.duration_seconds,
            time_data.duration_seconds.sum()
            .over(
                ibis.range_window(
                    group_by=[time_data.person, time_data.team],
                    order_by=[time_data.start_time, time_data.end_time],
                    following=0,
                )
            )
            .name("running_total"),
            time_data.duration_seconds.count()
            .over(
                ibis.range_window(
                    group_by=time_data.person, order_by=count_column, following=0
                )
            )
            .name("running_count"),
            time_data.duration_seconds.mean()
            .over(
                ibis.range_window(
                    group_by=time_data.person, order_by=count_column, following=0
                )
            )
            .name("running_avg"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


window_frame_params = pytest.mark.parametrize(
    "sql_window,window_args",
    [
        ("UNBOUNDED PRECEDING", {"preceding": None, "following": 0}),
        (
            "BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
            {"preceding": None, "following": None},
        ),
        (
            "BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
            {"preceding": 0, "following": None},
        ),
        ("5 PRECEDING", {"preceding": 5, "following": 0}),
        ("BETWEEN 10 PRECEDING AND 10 FOLLOWING", {"preceding": 10, "following": 10}),
    ],
)


@assert_state_not_change
@window_frame_params
def test_window_rows(time_data, sql_window, window_args):
    my_table = query(
        f"""SELECT count,
       duration_seconds,
       SUM(duration_seconds) OVER
         (ORDER BY start_time ROWS {sql_window}) AS running_total
  FROM time_data"""
    )
    ibis_table = time_data.projection(
        [
            time_data.get_column("count"),
            time_data.duration_seconds,
            time_data.duration_seconds.sum()
            .over(ibis.window(order_by=time_data.start_time, **window_args))
            .name("running_total"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
@window_frame_params
def test_window_range(time_data, sql_window, window_args):
    my_table = query(
        f"""SELECT count,
       duration_seconds,
       SUM(duration_seconds) OVER
         (ORDER BY start_time RANGE {sql_window}) AS running_total
  FROM time_data"""
    )
    ibis_table = time_data.projection(
        [
            time_data.get_column("count"),
            time_data.duration_seconds,
            time_data.duration_seconds.sum()
            .over(ibis.range_window(order_by=time_data.start_time, **window_args))
            .name("running_total"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


def test_multi_column_joins(time_data):
    my_table = query(
        """
    SELECT
        table1.team,
        table1.start_time_count,
        table2.start_time_count_d
    FROM
    (SELECT
        team,
        count(start_time)
        AS start_time_count
    FROM
        time_data
        GROUP BY team) table1
        INNER JOIN
    (SELECT team, count(start_time) AS start_time_count_d FROM
        (SELECT distinct team, start_time FROM time_data) intermediate GROUP BY team
    ) table2
        ON
            table1.team = table2.team AND
            table1.start_time_count = table2.start_time_count_d
    """
    )
    table1 = time_data.group_by(time_data.team).aggregate(
        [time_data.start_time.count().name("start_time_count")]
    )
    intermediate = time_data.projection(
        [time_data.team, time_data.start_time]
    ).distinct()
    table2 = intermediate.group_by(intermediate.team).aggregate(
        [intermediate.start_time.count().name("start_time_count_d")]
    )
    ibis_table = table1.join(
        table2,
        predicates=(
            (table1.team == table2.team)
            & (table1.start_time_count == table2.start_time_count_d)
        ),
        how="inner",
    ).projection([table1.team, table1.start_time_count, table2.start_time_count_d])
    assert_ibis_equal_show_diff(ibis_table, my_table)


def test_select_star_with_table_specified(time_data):
    my_table = query("select time_data.* from time_data")
    ibis_table = time_data
    assert_ibis_equal_show_diff(ibis_table, my_table)


def test_filter_on_non_selected_column(forest_fires):
    my_table = query("select temp from forest_fires where month = 'mar'")
    ibis_table = forest_fires[forest_fires.month == "mar"].projection(
        [forest_fires.temp]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)
