import ibis
from pandas import read_csv
from pandas.core.frame import DataFrame
import pytest

from sql_to_ibis import register_temp_table, remove_temp_table
from sql_to_ibis.tests.utils import DATA_PATH, get_all_join_columns_handle_duplicates

scope_fixture = pytest.fixture(scope="session")


@scope_fixture
def pandas_client():
    return ibis.pandas.PandasClient({})


@scope_fixture
def digimon_mon_list(pandas_client):
    frame = read_csv(DATA_PATH / "DigiDB_digimonlist.csv")
    frame["mon_attribute"] = frame["Attribute"]
    return ibis.pandas.from_dataframe(
        frame,
        "DIGIMON_MON_LIST",
        pandas_client,
    )


@scope_fixture
def digimon_move_list(pandas_client):
    frame = read_csv(DATA_PATH / "DigiDB_movelist.csv")
    frame["move_attribute"] = frame["Attribute"]
    return ibis.pandas.from_dataframe(frame, "DIGIMON_MOVE_LIST", pandas_client)


@scope_fixture
def forest_fires(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "forestfires.csv"), "FOREST_FIRES", pandas_client
    )


@scope_fixture
def avocado(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "avocado.csv"), "AVOCADO", pandas_client
    )


@scope_fixture
def time_data(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "time_data.csv"), "TIME_DATA", pandas_client
    )


@scope_fixture
def multitable_join_main_table():
    return ibis.pandas.from_dataframe(
        DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "lookup_id": [1, 5, 8, 9, 10],
                "relationship_id": [0, 1, 2, 2, 1],
                "promotion_id": [0, 1, 2, 1, 0],
            }
        )
    )


@scope_fixture
def multitable_join_lookup_table():
    return ibis.pandas.from_dataframe(
        DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "value": [0, 3, 20, 10, 40, 20, 10, 10, 10, 10],
            }
        )
    )


@scope_fixture
def multitable_join_relationship_table():
    return ibis.pandas.from_dataframe(
        DataFrame({"id": [0, 1, 2], "relation": ["rel1", "rel2", "rel3"]})
    )


@scope_fixture
def multitable_join_promotion_table():
    return ibis.pandas.from_dataframe(
        DataFrame({"id": [0, 1, 2], "promotion": ["none", "special", "extra special"]})
    )


@pytest.fixture(autouse=True, scope="session")
def register_temp_tables(
    digimon_mon_list,
    digimon_move_list,
    forest_fires,
    time_data,
    avocado,
    multitable_join_main_table,
    multitable_join_lookup_table,
    multitable_join_relationship_table,
    multitable_join_promotion_table,
):
    tables = {
        "DIGIMON_MON_LIST": digimon_mon_list,
        "DIGIMON_MOVE_LIST": digimon_move_list,
        "FOREST_FIRES": forest_fires,
        "TIME_DATA": time_data,
        "AVOCADO": avocado,
        "MULTI_MAIN": multitable_join_main_table,
        "MULTI_LOOKUP": multitable_join_lookup_table,
        "MULTI_RELATIONSHIP": multitable_join_relationship_table,
        "MULTI_PROMOTION": multitable_join_promotion_table,
    }
    for table_name in tables:
        register_temp_table(tables[table_name], table_name)
    yield
    for table_name in tables:
        remove_temp_table(table_name)


@pytest.fixture
def digimon_move_mon_join_columns(digimon_mon_list, digimon_move_list):
    return get_all_join_columns_handle_duplicates(
        digimon_mon_list, digimon_move_list, "DIGIMON_MON_LIST", "DIGIMON_MOVE_LIST"
    )
