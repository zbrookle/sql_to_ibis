from datetime import date, datetime
from typing import Callable

from freezegun import freeze_time
import ibis
import pytest

from sql_to_ibis import query
from sql_to_ibis.tests.utils import assert_ibis_equal_show_diff, assert_state_not_change


@assert_state_not_change
def test_type_conversion(forest_fires):
    """
    Tests sql as statements
    :return:
    """
    my_table = query(
        """select cast(temp as int64),
        cast(RH as float64) my_rh,
        wind,
        rain,
        area,
        cast(2.0 as int64) my_int,
        cast(3 as float64) as my_float,
        cast(7 as object) as my_object,
        cast(0 as bool) as my_bool from forest_fires"""
    )
    fire_frame = forest_fires.projection(
        [
            forest_fires.temp.cast("int64").name("temp"),
            forest_fires.RH.cast("float64").name("my_rh"),
            forest_fires.wind,
            forest_fires.rain,
            forest_fires.area,
            ibis.literal(2.0).cast("int64").name("my_int"),
            ibis.literal(3).cast("float64").name("my_float"),
            ibis.literal(7).cast("string").name("my_object"),
            ibis.literal(0).cast("bool").name("my_bool"),
        ]
    )
    assert_ibis_equal_show_diff(my_table, fire_frame)


@assert_state_not_change
@pytest.mark.parametrize("string", ["mar ", " mar", " mar ", "m ar", " ", ""])
def test_string_spaces(forest_fires, string: str):
    my_table = query(f"""select * from forest_fires where month = '{string}'""")
    ibis_table = forest_fires[forest_fires.month == string]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
@pytest.mark.parametrize(
    "query_null,ibis_bool_func",
    [
        ("", lambda ff_table: ff_table.temp == ibis.null()),
        ("not", lambda ff_table: ff_table.temp != ibis.null()),
    ],
)
def test_null(forest_fires, query_null: str, ibis_bool_func: Callable):
    my_table = query(f"select * from forest_fires where temp is {query_null} null")
    ibis_table = forest_fires[ibis_bool_func(forest_fires)]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_set_string_value_as_column_value(forest_fires):
    """
    Select a string like 'Yes' as a column value
    :return:
    """
    my_table = query(
        """
    select wind, 'yes' as wind_yes from forest_fires"""
    )
    ibis_table = forest_fires[["wind"]].mutate(ibis.literal("yes").name("wind_yes"))
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_datetime_cast(forest_fires):
    """
    Select casting a string as a date
    :return:
    """
    my_table = query(
        """
    select wind, cast('2019-01-01' as datetime64) as my_date from forest_fires"""
    )
    ibis_table = forest_fires[["wind"]].mutate(
        ibis.literal("2019-01-01").cast("timestamp").name("my_date")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_date_cast(forest_fires):
    """
    Select casting a string as a date
    :return:
    """
    my_table = query(
        """
    select wind, cast('2019-01-01' as date) as my_date from forest_fires"""
    )
    ibis_table = forest_fires[["wind"]].mutate(
        ibis.literal("2019-01-01").cast("date").name("my_date")
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_timestamps(forest_fires):
    """
    Select now() as date
    :return:
    """
    with freeze_time(datetime.now()):
        my_table = query(
            """
        select wind, now(), today(), timestamp('2019-01-31', '23:20:32')
        from forest_fires"""
        )
        ibis_table = forest_fires[["wind"]].mutate(
            [
                ibis.literal(datetime.now()).name("now()"),
                ibis.literal(date.today()).name("today()"),
                ibis.literal(datetime(2019, 1, 31, 23, 20, 32)).name("_literal2"),
            ]
        )
        assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_sql_data_types(avocado):
    """
    Tests sql data types
    :return:
    """
    my_table = query(
        """
        select
            cast(avocado_id as object) as avocado_id_object,
            cast(avocado_id as int16) as avocado_id_int16,
            cast(avocado_id as smallint) as avocado_id_smallint,
            cast(avocado_id as int32) as avocado_id_int32,
            cast(avocado_id as int) as avocado_id_int,
            cast(avocado_id as integer) as avocado_id_integer,
            cast(avocado_id as int64) as avocado_id_int64,
            cast(avocado_id as bigint) as avocado_id_bigint,
            cast(avocado_id as float) as avocado_id_float,
            cast(avocado_id as float16) as avocado_id_float16,
            cast(avocado_id as float32) as avocado_id_float32,
            cast(avocado_id as float64) as avocado_id_float64,
            cast(avocado_id as bool) as avocado_id_bool,
            cast(avocado_id as category) as avocado_id_category,
            cast(date as date) as date,
            cast(date as datetime64) as datetime,
            cast(date as timestamp) as timestamp,
            cast(date as time) as time,
            cast(region as varchar) as region_varchar,
            cast(region as string) as region_string
        from avocado
        """
    )

    date_column = avocado.Date
    id_column = avocado.avocado_id
    region_column = avocado.region
    ibis_table = avocado.projection(
        [
            id_column.cast("string").name("avocado_id_object"),
            id_column.cast("int16").name("avocado_id_int16"),
            id_column.cast("int16").name("avocado_id_smallint"),
            id_column.cast("int32").name("avocado_id_int32"),
            id_column.cast("int32").name("avocado_id_int"),
            id_column.cast("int32").name("avocado_id_integer"),
            id_column.cast("int64").name("avocado_id_int64"),
            id_column.cast("int64").name("avocado_id_bigint"),
            id_column.cast("float").name("avocado_id_float"),
            id_column.cast("float16").name("avocado_id_float16"),
            id_column.cast("float32").name("avocado_id_float32"),
            id_column.cast("float64").name("avocado_id_float64"),
            id_column.cast("bool").name("avocado_id_bool"),
            id_column.cast("category").name("avocado_id_category"),
            date_column.cast("date").name("date"),
            date_column.cast("timestamp").name("datetime"),
            date_column.cast("timestamp").name("timestamp"),
            date_column.cast("time").name("time"),
            region_column.cast("string").name("region_varchar"),
            region_column.cast("string").name("region_string"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)
