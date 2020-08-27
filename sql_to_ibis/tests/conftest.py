import ibis
from pandas import read_csv
import pytest

from sql_to_ibis import register_temp_table, remove_temp_table
from sql_to_ibis.tests.utils import DATA_PATH


@pytest.fixture(scope="session")
def pandas_client():
    return ibis.pandas.PandasClient({})


@pytest.fixture(scope="session")
def digimon_mon_list(pandas_client):
    frame = read_csv(DATA_PATH / "DigiDB_digimonlist.csv")
    frame["mon_attribute"] = frame["Attribute"]
    return ibis.pandas.from_dataframe(
        frame,
        "DIGIMON_MON_LIST",
        pandas_client,
    )


@pytest.fixture(scope="session")
def digimon_move_list(pandas_client):
    frame = read_csv(DATA_PATH / "DigiDB_movelist.csv")
    frame["move_attribute"] = frame["Attribute"]
    return ibis.pandas.from_dataframe(frame, "DIGIMON_MOVE_LIST", pandas_client)


@pytest.fixture(scope="session")
def forest_fires(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "forestfires.csv"), "FOREST_FIRES", pandas_client
    )


@pytest.fixture(scope="session")
def avocado(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "avocado.csv"), "AVOCADO", pandas_client
    )


@pytest.fixture(scope="session")
def time_data(pandas_client):
    return ibis.pandas.from_dataframe(
        read_csv(DATA_PATH / "time_data.csv"), "TIME_DATA", pandas_client
    )


@pytest.fixture(autouse=True, scope="session")
def register_temp_tables(
    digimon_mon_list, digimon_move_list, forest_fires, time_data, avocado
):
    tables = {
        "DIGIMON_MON_LIST": digimon_mon_list,
        "DIGIMON_MOVE_LIST": digimon_move_list,
        "FOREST_FIRES": forest_fires,
        "TIME_DATA": time_data,
        "AVOCADO": avocado,
    }
    for table_name in tables:
        register_temp_table(tables[table_name], table_name)
    yield
    for table_name in tables:
        remove_temp_table(table_name)
