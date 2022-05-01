import ibis
from ibis.expr.types import TableExpr

from sql_to_ibis import query
from sql_to_ibis.tests.utils import assert_ibis_equal_show_diff, assert_state_not_change


@assert_state_not_change
def test_coalesce(forest_fires: TableExpr) -> None:
    my_table = query("select coalesce(wind, rain, 2) as my_number from forest_fires")
    ibis_table = forest_fires[
        ibis.coalesce(forest_fires.wind, forest_fires.rain, 2).name("my_number")
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)
