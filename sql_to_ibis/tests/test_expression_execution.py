import ibis
from ibis.expr.api import TableExpr
from pandas import read_csv
from pandas.testing import assert_frame_equal
import pytest

from sql_to_ibis import query, register_temp_table, remove_temp_table
from sql_to_ibis.tests.utils import (
    DATA_PATH,
    get_all_join_columns_handle_duplicates,
    get_columns_with_alias,
    join_params,
)


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


@pytest.fixture(scope="module")
def forest_fires(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "forestfires.csv"), "FOREST_FIRES", pandas_client
    )


@pytest.fixture(autouse=True, scope="module")
def register_temp_tables(digimon_mon_list, digimon_move_list, forest_fires):
    register_temp_table(digimon_mon_list, "DIGIMON_MON_LIST")
    register_temp_table(digimon_move_list, "DIGIMON_MOVE_LIST")
    register_temp_table(forest_fires, "FOREST_FIRES")
    yield
    for table in ["DIGIMON_MON_LIST", "DIGIMON_MOVE_LIST", "FOREST_FIRES"]:
        remove_temp_table(table)


@pytest.fixture
def digimon_move_mon_join_columns(digimon_mon_list, digimon_move_list):
    return get_all_join_columns_handle_duplicates(
        digimon_mon_list, digimon_move_list, "DIGIMON_MON_LIST", "DIGIMON_MOVE_LIST"
    )


@join_params
def test_select_star_join_execution(
    pandas_client,
    sql_join,
    ibis_join,
    digimon_mon_list: TableExpr,
    digimon_move_list: TableExpr,
    digimon_move_mon_join_columns,
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


def test_agg_with_group_by_with_select_groupby_execution(forest_fires):
    my_frame = query(
        "select day, month, min(temp), max(temp) from forest_fires group by day, month"
    ).execute()
    ibis_frame = (
        forest_fires.groupby([forest_fires.day, forest_fires.month])
        .aggregate(
            [
                forest_fires.temp.min().name("_col0"),
                forest_fires.temp.max().name("_col1"),
            ]
        )
        .execute()
    )
    assert_frame_equal(ibis_frame, my_frame)


def test_agg_with_group_by_without_select_groupby_execution(forest_fires):
    my_frame = query(
        "select min(temp), max(temp) from forest_fires group by day, month"
    ).execute()
    ibis_frame = (
        forest_fires.groupby([forest_fires.day, forest_fires.month])
        .aggregate(
            [
                forest_fires.temp.min().name("_col0"),
                forest_fires.temp.max().name("_col1"),
            ]
        )
        .drop(["day", "month"])
        .execute()
    )
    assert_frame_equal(ibis_frame, my_frame)


def test_select_columns_from_two_tables_with_same_column_name(forest_fires):
    """
    Test selecting tables
    :return:
    """
    my_frame = query(
        """select * from forest_fires table1, forest_fires table2"""
    ).execute()
    ibis_frame = forest_fires.cross_join(forest_fires)[
        get_columns_with_alias(forest_fires, "table1")
        + get_columns_with_alias(forest_fires, "table2")
    ].execute()
    assert_frame_equal(ibis_frame, my_frame)


def test_select_star_from_multiple_tables(
    digimon_move_list, digimon_mon_list, digimon_move_mon_join_columns
):
    """
    Test selecting from two different tables
    :return:
    """
    my_frame = query("""select * from digimon_mon_list, digimon_move_list""").execute()
    ibis_frame = digimon_mon_list.cross_join(digimon_move_list)[
        digimon_move_mon_join_columns
    ].execute()
    assert_frame_equal(ibis_frame, my_frame)


# @pytest.mark.parametrize(
#     "sql",
#     [
#         """
#     select * from
#     ((select X, Y, rain from forest_fires) table1
#     join
#     (select X, Y, rain from forest_fires) table2
#     on table1.x = table2.x) sub
#     """,
#         """
#     select * from
#     (select X, Y, rain from forest_fires) table1
#     join
#     (select X, Y, rain from forest_fires) table2
#     on table1.x = table2.x
#     """,
#     ],
# )
# def test_joining_two_subqueries_with_overlapping_columns(sql, forest_fires):
#     # my_table = query(sql).execute()
#     columns = ["X", "Y", "rain"]
#
#     def get_select_rename_columns(alias: str):
#         my_columns = forest_fires.get_columns(columns)
#         renamed_columns = []
#         for i, column in enumerate(my_columns):
#             renamed_columns.append(column.name(f"{alias}.{columns[i]}"))
#         return my_columns, renamed_columns
#
#     select1, renamed1 = get_select_rename_columns("table1")
#     select2, renamed2 = get_select_rename_columns("table2")
#     subquery1 = forest_fires[select1]
#     subquery2 = forest_fires[select2]
#     print(subquery1)
#     print(subquery2)
#     joined = (
#         subquery1.join(subquery2, predicates=subquery1.X == subquery2.X)
#         .projection(renamed1 + renamed2)
#         .execute()
#     )
#     print(joined)
#
#     assert_frame_equal(joined, my_table)
