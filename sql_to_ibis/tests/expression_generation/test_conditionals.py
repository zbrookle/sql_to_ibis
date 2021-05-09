from typing import List

import ibis
from ibis.expr.operations import Literal
import pytest

from sql_to_ibis import query
from sql_to_ibis.tests.utils import assert_ibis_equal_show_diff, assert_state_not_change


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


in_list_params = pytest.mark.parametrize(
    "sql,ibis_expr_list",
    [
        (
            "('fri', 'sun')",
            [ibis.literal("fri"), ibis.literal("sun")],
        ),
        (
            "('fri', 'sun', 'sat')",
            [ibis.literal("fri"), ibis.literal("sun"), ibis.literal("sat")],
        ),
    ],
)


@assert_state_not_change
@in_list_params
def test_in_operator(forest_fires, sql: str, ibis_expr_list: List[Literal]):
    """
    Test using in operator in a sql query
    :return:
    """
    my_table = query(
        f"""
    select * from forest_fires where day in {sql}
    """
    )
    ibis_table = forest_fires[forest_fires.day.isin(ibis_expr_list)]
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
@in_list_params
def test_not_in_operator(forest_fires, sql: str, ibis_expr_list: List[Literal]):
    """
    Test using in operator in a sql query
    :return:
    """
    my_table = query(
        f"""
    select * from forest_fires where day not in {sql}
    """
    )
    ibis_table = forest_fires[forest_fires.day.notin(ibis_expr_list)]
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


@assert_state_not_change
def test_filter_on_non_selected_column(forest_fires):
    my_table = query("select temp from forest_fires where month = 'mar'")
    ibis_table = forest_fires[forest_fires.month == "mar"].projection(
        [forest_fires.temp]
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
