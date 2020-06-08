from sql_to_ibis import register_temp_table, query, remove_temp_table
from sql_to_ibis.tests.utils import DATA_PATH, join_params, get_all_join_columns_handle_duplicates
import pytest
from ibis.expr.api import TableExpr
from pandas.testing import assert_frame_equal

from pandas import read_csv

import ibis


@pytest.fixture(scope="module")
def pandas_client():
    return ibis.pandas.PandasClient({})


@pytest.fixture(scope="module")
def digimon_mon_list(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "DigiDB_digimonlist.csv"),
        "DIGIMON_MON_LIST",
        pandas_client,
    )


@pytest.fixture(scope="module")
def digimon_move_list(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "DigiDB_movelist.csv"), "DIGIMON_MOVE_LIST", pandas_client
    )


@pytest.fixture(autouse=True, scope="module")
def register_temp_tables(digimon_mon_list, digimon_move_list):
    register_temp_table(digimon_mon_list, "DIGIMON_MON_LIST")
    register_temp_table(digimon_move_list, "DIGIMON_MOVE_LIST")
    yield
    for table in ["DIGIMON_MON_LIST", "DIGIMON_MOVE_LIST"]:
        remove_temp_table(table)

@pytest.fixture
def digimon_move_mon_join_columns(digimon_mon_list, digimon_move_list):
    return get_all_join_columns_handle_duplicates(
        digimon_mon_list, digimon_move_list, "DIGIMON_MON_LIST", "DIGIMON_MOVE_LIST"
    )


@join_params
def test_join_execution(
    pandas_client,
    sql_join,
    ibis_join,
    digimon_mon_list: TableExpr,
    digimon_move_list: TableExpr,
    digimon_move_mon_join_columns
):
    my_frame = query(
        f"""select * from digimon_mon_list {sql_join} join
            digimon_move_list
            on digimon_mon_list.attribute = digimon_move_list.attribute"""
    ).execute()
    ibis_frame = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_mon_list.Attribute == digimon_move_list.Attribute,
        how=ibis_join,
    )[digimon_move_mon_join_columns].execute()
    assert_frame_equal(ibis_frame, my_frame)
