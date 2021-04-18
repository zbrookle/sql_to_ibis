import ibis

from sql_to_ibis import query
from sql_to_ibis.tests.utils import assert_ibis_equal_show_diff, assert_state_not_change


@assert_state_not_change
def test_coalesce(forest_fires):
    my_table = query("select coalesce(day, month, 2) as my_number from forest_fires")
    ibis_table = ibis.coalesce(9, forest_fires.day, forest_fires.month).name(
        "my_number"
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)
