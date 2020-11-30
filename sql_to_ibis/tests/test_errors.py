import pytest

from sql_to_ibis import query
from sql_to_ibis.exceptions.sql_exception import (
    AmbiguousColumnException,
    ColumnNotFoundError,
    InvalidQueryException,
    TableExprDoesNotExist,
    UnsupportedColumnOperation,
)
import sql_to_ibis.sql.sql_value_objects
from sql_to_ibis.tests.utils import assert_state_not_change


# TODO Make a session object so that class variables don't need to be reset
@pytest.mark.parametrize(
    "sql",
    [
        "hello world!",
        "select type from digimon_move_list having max(power) > 40",
        """select * from digimon_mon_list cross join
         digimon_move_list
         on digimon_mon_list.type = digimon_move_list.type""",
        """select move, type, power from
            digimon_move_list
            where
            power in
            ( select max(power) as power, type
             from digimon_move_list
             group by type ) t1""",
    ],
)
@assert_state_not_change
def test_invalid_queries(sql):
    with pytest.raises(InvalidQueryException):
        query(sql)
    sql_to_ibis.sql.sql_value_objects.DerivedColumn.reset_expression_count()
    sql_to_ibis.sql.sql_value_objects.Literal.reset_literal_count()


@pytest.mark.parametrize(
    "sql",
    [
        "select not_here from forest_fires",
        "select * from digimon_mon_list join "
        "digimon_move_list on "
        "digimon_mon_list.not_here = digimon_move_list.attribute",
        "select forest_fires.not_here from forest_fires",
    ],
)
def test_raise_error_for_choosing_column_not_in_table(sql: str):
    with pytest.raises(ColumnNotFoundError):
        query(sql)


@assert_state_not_change
def test_for_non_existent_table():
    """
    Check that exception is raised if table does not exist
    :return:
    """
    with pytest.raises(TableExprDoesNotExist):
        query("select * from a_table_that_is_not_here")


@assert_state_not_change
def test_ambiguous_column():
    with pytest.raises(AmbiguousColumnException):
        query("select type from digimon_move_list, digimon_mon_list")


@assert_state_not_change
def test_unsupported_operation_exception():
    with pytest.raises(UnsupportedColumnOperation):
        query("select sum(month) from forest_fires")
