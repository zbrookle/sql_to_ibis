from sql_to_ibis import query
from sql_to_ibis.tests.markers import ibis_not_implemented
from sql_to_ibis.tests.utils import (
    assert_ibis_equal_show_diff,
    assert_state_not_change,
    resolved_columns,
)


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
def test_count_star(forest_fires):
    my_table = query("select count(*) from forest_fires")
    ibis_table = forest_fires.aggregate([forest_fires.count().name("_col0")])
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_count_star_cross_join(digimon_move_list, digimon_mon_list):
    my_table = query(
        "select count(*) from digimon_move_list cross join digimon_mon_list"
    )
    resolved = resolved_columns(
        digimon_move_list, digimon_mon_list, "DIGIMON_MOVE_LIST", "DIGIMON_MON_LIST"
    )
    cross_join_table = digimon_move_list.cross_join(digimon_mon_list).projection(
        resolved
    )
    ibis_table = cross_join_table.aggregate([cross_join_table.count().name("_col0")])
    assert_ibis_equal_show_diff(ibis_table, my_table)


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
def test_count_distinct(digimon_move_list):
    my_table = query("select count(distinct type) from digimon_move_list")
    ibis_table = digimon_move_list.aggregate(
        digimon_move_list.Type.nunique().name("_col0")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)
