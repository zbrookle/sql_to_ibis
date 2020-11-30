from sql_to_ibis import register_temp_table, remove_temp_table
from sql_to_ibis.sql.sql_objects import AmbiguousColumn
from sql_to_ibis.sql_select_query import TableInfo
from sql_to_ibis.tests.utils import assert_ibis_equal_show_diff


def test_add_remove_temp_table(digimon_mon_list):
    """
    Tests registering and removing temp tables
    :return:
    """
    frame_name = "digimon_mon_list"
    real_frame_name = TableInfo.ibis_table_name_map[frame_name]
    remove_temp_table(frame_name)
    tables_present_in_column_to_dataframe = set()
    for column in TableInfo.column_to_table_name:
        table = TableInfo.column_to_table_name[column]
        if isinstance(table, AmbiguousColumn):
            for table_name in table.tables:
                tables_present_in_column_to_dataframe.add(table_name)
        else:
            tables_present_in_column_to_dataframe.add(table)

    # Ensure column metadata is removed correctly
    assert (
        frame_name not in TableInfo.ibis_table_name_map
        and real_frame_name not in TableInfo.ibis_table_map
        and real_frame_name not in TableInfo.column_name_map
        and real_frame_name not in tables_present_in_column_to_dataframe
    )

    registered_frame_name = real_frame_name
    register_temp_table(digimon_mon_list, registered_frame_name)

    assert (
        TableInfo.ibis_table_name_map.get(frame_name.lower()) == registered_frame_name
        and real_frame_name in TableInfo.column_name_map
    )

    assert_ibis_equal_show_diff(
        TableInfo.ibis_table_map[registered_frame_name].get_table_expr(),
        digimon_mon_list,
    )

    # Ensure column metadata is added correctly
    for column in digimon_mon_list.columns:
        assert column == TableInfo.column_name_map[registered_frame_name].get(
            column.lower()
        )
        lower_column = column.lower()
        assert lower_column in TableInfo.column_to_table_name
        table = TableInfo.column_to_table_name.get(lower_column)
        if isinstance(table, AmbiguousColumn):
            assert registered_frame_name in table.tables
        else:
            assert registered_frame_name == table
