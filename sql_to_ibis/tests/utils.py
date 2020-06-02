"""
Shared functions among the tests like setting up test environment
"""
from copy import deepcopy
from functools import wraps
from pathlib import Path
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile
from typing import Callable

import ibis
from ibis.expr.api import TableExpr
from ibis.tests.util import assert_equal
from pandas import DataFrame, read_csv

from sql_to_ibis import register_temp_table, remove_temp_table
from sql_to_ibis.sql_select_query import TableInfo

DATA_PATH = Path(__file__).parent.parent / "data"


def pandas_to_ibis(frame: DataFrame, name: str):
    tuples = []
    for column in frame:
        dtype = str(frame[column].dtype)
        if dtype == "object":
            dtype = "string"
        tuples.append((frame[column].name, dtype))
    test = ibis.table(ibis.schema(tuples), name)
    return test


# Import the data for testing
FOREST_FIRES: TableExpr = pandas_to_ibis(
    read_csv(DATA_PATH / "forestfires.csv"), "FOREST_FIRES"
)
DIGIMON_MON_LIST = read_csv(DATA_PATH / "DigiDB_digimonlist.csv")
DIGIMON_MOVE_LIST = read_csv(DATA_PATH / "DigiDB_movelist.csv")
DIGIMON_SUPPORT_LIST: TableExpr = pandas_to_ibis(
    read_csv(DATA_PATH / "DigiDB_supportlist.csv"), "DIGIMON_SUPPORT_LIST"
)
AVOCADO: TableExpr = pandas_to_ibis(read_csv(DATA_PATH / "avocado.csv"), "AVOCADO")

# Name change is for name interference
DIGIMON_MON_LIST["mon_attribute"] = DIGIMON_MON_LIST["Attribute"]
DIGIMON_MOVE_LIST["move_attribute"] = DIGIMON_MOVE_LIST["Attribute"]
DIGIMON_MON_LIST = pandas_to_ibis(DIGIMON_MON_LIST, "DIGIMON_MON_LIST")
DIGIMON_MOVE_LIST = pandas_to_ibis(DIGIMON_MOVE_LIST, "DIGIMON_MOVE_LIST")


def register_env_tables():
    """
    Returns all globals but in lower case
    :return:
    """
    for variable_name in globals():
        variable = globals()[variable_name]
        if isinstance(variable, TableExpr):
            register_temp_table(table=variable, table_name=variable_name)


def remove_env_tables():
    """
    Remove all env tables
    :return:
    """
    for variable_name in globals():
        variable = globals()[variable_name]
        if isinstance(variable, DataFrame):
            remove_temp_table(table_name=variable_name)


def display_dict_difference(before_dict: dict, after_dict: dict, name: str):
    dict_diff_report = f"Dictionary Difference Report for {name}:\n"
    for key in before_dict:
        after_value = after_dict.get(key)
        before_value = before_dict[key]
        if after_value != before_value:
            dict_diff_report += (
                f"Value at key '{key}' was {before_value} and now is "
                f"{after_value}\n"
            )
    for key in after_dict:
        if before_dict.get(key) is None:
            dict_diff_report += (
                f"There is now a value {after_dict[key]} at '{key}', "
                f"but there was nothing there before\n"
            )

    raise Exception(dict_diff_report)


def assert_state_not_change(func: Callable):
    @wraps(func)
    def new_func():
        table_state = {}
        for key in TableInfo.ibis_table_map:
            table_state[key] = TableInfo.ibis_table_map[key]
        column_to_dataframe_name = deepcopy(TableInfo.column_to_dataframe_name)
        column_name_map = deepcopy(TableInfo.column_name_map)
        dataframe_name_map = deepcopy(TableInfo.ibis_table_name_map)

        func()

        for key in TableInfo.ibis_table_map:
            assert table_state[key] == TableInfo.ibis_table_map[key]
        if column_to_dataframe_name != TableInfo.column_to_dataframe_name:
            display_dict_difference(
                column_to_dataframe_name,
                TableInfo.column_to_dataframe_name,
                "column_to_dataframe_name",
            )
        if column_name_map != TableInfo.column_name_map:
            display_dict_difference(
                column_name_map, TableInfo.column_name_map, "column_name_map"
            )
        if dataframe_name_map != TableInfo.ibis_table_name_map:
            display_dict_difference(
                dataframe_name_map, TableInfo.ibis_table_name_map, "dataframe_name_map"
            )

    return new_func


def assert_ibis_equal_show_diff(obj1: TableExpr, obj2: TableExpr):
    if not isinstance(obj1, TableExpr):
        raise AssertionError(f"{obj1} is not of type TableExpr")
    if not isinstance(obj2, TableExpr):
        raise AssertionError(f"{obj2} is not of type TableExpr")
    try:
        assert_equal(obj1, obj2)
    except AssertionError:
        obj1_str = str(obj1)
        obj2_str = str(obj2)
        if obj1_str != obj2_str:
            with NamedTemporaryFile(delete=False) as obj1_file, NamedTemporaryFile(
                delete=False
            ) as obj2_file:
                obj1_file.write(bytes(obj1_str, encoding="utf-8"))
                obj1_file.close()
                obj2_file.write(bytes(obj2_str, encoding="utf-8"))
                obj2_file.close()
                process = Popen(
                    ["diff", "-y", obj1_file.name, obj2_file.name],
                    stdout=PIPE,
                    stderr=PIPE,
                )
                output, _ = process.communicate()
                str_output = output.decode("utf-8")

            msg = f"Plan representations not equal!\n{str_output}"
            raise AssertionError(msg)
