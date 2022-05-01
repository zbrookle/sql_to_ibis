from typing import Dict, Optional, Tuple

import ibis
from ibis.expr.types import TableExpr
import pytest

from sql_to_ibis import query
from sql_to_ibis.tests.utils import assert_ibis_equal_show_diff, assert_state_not_change


# TODO Ibis is showing window as object in string representation
@assert_state_not_change
def test_rank_statement_one_column(forest_fires: TableExpr) -> None:
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
def test_rank_statement_many_columns(forest_fires: TableExpr) -> None:
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
def test_dense_rank_statement_many_columns(forest_fires: TableExpr) -> None:
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
def test_rank_over_partition_by(forest_fires: TableExpr) -> None:
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
def test_partition_by_multiple_columns(forest_fires: TableExpr) -> None:
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
def test_dense_rank_over_partition_by(forest_fires: TableExpr) -> None:
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
def test_window_function_partition(time_data: TableExpr) -> None:
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
def test_window_function_partition_order_by(time_data: TableExpr) -> None:
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
def test_window_rows(
    time_data: TableExpr,
    sql_window: str,
    window_args: Tuple[str, Dict[str, Optional[int]]],
) -> None:
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
def test_window_range(
    time_data: TableExpr,
    sql_window: str,
    window_args: Tuple[str, Dict[str, Optional[int]]],
) -> None:
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


@assert_state_not_change
def test_order_by(forest_fires: TableExpr) -> None:
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
def test_order_by_case_insensitive(forest_fires: TableExpr) -> None:
    """
    Test case sensitivity in order by clause
    """
    my_table = query(
        """select * from forest_fires order by TeMp desc, WIND asc, areA"""
    )
    ibis_table = forest_fires.sort_by(
        [
            (forest_fires.temp, False),
            (forest_fires.wind, True),
            (forest_fires.area, True),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)
