"""
Test cases for panda to sql
"""


import ibis

from sql_to_ibis import query
from sql_to_ibis.tests.utils import assert_ibis_equal_show_diff, assert_state_not_change


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
