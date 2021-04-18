from ibis.expr.types import TableExpr
import pytest

from sql_to_ibis import query
from sql_to_ibis.tests.utils import (
    assert_ibis_equal_show_diff,
    assert_state_not_change,
    join_params,
    resolved_columns,
)


@assert_state_not_change
def test_distinct(forest_fires):
    """
    Test use of the distinct keyword
    :return:
    """
    my_table = query("select distinct area, rain from forest_fires")
    ibis_table = forest_fires[["area", "rain"]].distinct()
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_columns_maintain_order_chosen(forest_fires):
    my_table = query("select area, rain from forest_fires")
    ibis_table = forest_fires[["area", "rain"]]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_subquery(forest_fires):
    """
    Test ability to perform subqueries
    :return:
    """
    my_table = query("select * from (select area, rain from forest_fires) rain_area")
    ibis_table = forest_fires[["area", "rain"]]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@join_params
@assert_state_not_change
def test_joins(
    digimon_move_mon_join_columns,
    sql_join: str,
    ibis_join: str,
    digimon_move_list,
    digimon_mon_list,
):
    my_table = query(
        f"select * from digimon_mon_list {sql_join} join "
        "digimon_move_list on "
        "digimon_mon_list.attribute = digimon_move_list.attribute"
    )
    ibis_table = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_mon_list.Attribute == digimon_move_list.Attribute,
        how=ibis_join,
    )[digimon_move_mon_join_columns]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@join_params
@assert_state_not_change
def test_join_specify_selection(
    sql_join: str, ibis_join: str, digimon_move_list, digimon_mon_list
):
    """
    Test join
    :return:
    """
    my_table = query(
        f"""select power from digimon_mon_list {sql_join} join
            digimon_move_list
            on digimon_mon_list.attribute = digimon_move_list.attribute"""
    )
    ibis_table = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_mon_list.Attribute == digimon_move_list.Attribute,
        how=ibis_join,
    )[digimon_move_list.Power.name("power")]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@join_params
@assert_state_not_change
def test_join_wo_specifying_table(
    digimon_move_mon_join_columns,
    sql_join: str,
    ibis_join: str,
    digimon_move_list,
    digimon_mon_list,
):
    """
    Test join where table isn't specified in join
    :return:
    """
    my_table = query(
        f"""
        select * from digimon_mon_list {sql_join} join
        digimon_move_list
        on mon_attribute = move_attribute
        """
    )
    ibis_table = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_mon_list.mon_attribute == digimon_move_list.move_attribute,
        how=ibis_join,
    )[digimon_move_mon_join_columns]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_cross_joins(
    digimon_move_mon_join_columns, digimon_move_list, digimon_mon_list
):
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_table = query(
        """select * from digimon_mon_list cross join
            digimon_move_list"""
    )
    ibis_table = digimon_mon_list.cross_join(digimon_move_list)[
        digimon_move_mon_join_columns
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_cross_join_with_selection(digimon_move_list, digimon_mon_list):
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_table = query(
        """select power from digimon_mon_list cross join
            digimon_move_list"""
    )
    ibis_table = digimon_mon_list.cross_join(digimon_move_list)[
        digimon_move_list.Power.name("power")
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_nested_subquery(forest_fires):
    """
    Test nested subqueries
    :return:
    """
    my_table = query(
        """select * from
            (select wind, rh from
              (select * from forest_fires) fires) wind_rh"""
    )
    ibis_table = forest_fires[["wind", "RH"]].relabel({"RH": "rh"})
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_union(forest_fires):
    """
    Test union in queries
    :return:
    """
    my_table = query(
        """
    select * from forest_fires order by wind desc limit 5
    union
    select * from forest_fires order by wind asc limit 5
    """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(5)
    ibis_table = ibis_table1.union(ibis_table2, distinct=True)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_union_distinct(forest_fires):
    """
    Test union distinct in queries
    :return:
    """
    my_table = query(
        """
        select * from forest_fires order by wind desc limit 5
         union distinct
        select * from forest_fires order by wind asc limit 5
        """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(5)
    ibis_table = ibis_table1.union(ibis_table2, distinct=True)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_union_all(forest_fires):
    """
    Test union distinct in queries
    :return:
    """
    my_table = query(
        """
        select * from forest_fires order by wind desc limit 5
         union all
        select * from forest_fires order by wind asc limit 5
        """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(5)
    ibis_table = ibis_table1.union(ibis_table2)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
@pytest.mark.parametrize("set_op", ["intersect", "intersect distinct"])
def test_intersect_distinct(forest_fires, set_op: str):
    """
    Test intersect in queries
    :return:
    """
    my_table = query(
        f"""
            select * from forest_fires order by wind desc limit 5
             {set_op}
            select * from forest_fires order by wind asc limit 3
            """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(3)
    ibis_table = ibis_table1.intersect(ibis_table2)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_except_distinct(forest_fires):
    """
    Test except distinct in queries
    :return:
    """
    my_table = query(
        """
                select * from forest_fires order by wind desc limit 5
                 except distinct
                select * from forest_fires order by wind asc limit 3
                """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(3)
    ibis_table = ibis_table1.difference(ibis_table2).distinct()
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_except_all(forest_fires):
    """
    Test except distinct in queries
    :return:
    """
    my_table = query(
        """
                select * from forest_fires order by wind desc limit 5
                 except all
                select * from forest_fires order by wind asc limit 3
                """
    )
    ibis_table1 = forest_fires.sort_by(("wind", False)).head(5)
    ibis_table2 = forest_fires.sort_by(("wind", True)).head(3)
    ibis_table = ibis_table1.difference(ibis_table2)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@pytest.mark.xfail(reason="This needs to be solved", raises=AssertionError)
@pytest.mark.parametrize(
    "sql",
    [
        """
    select * from
    (
    (select X, Y, rain from forest_fires) table1
    join
    (select X, Y, rain from forest_fires) table2
    on table1.x = table2.x) sub
    """,
        """
    select * from
    (select X, Y, rain from forest_fires) table1
    join
    (select X, Y, rain from forest_fires) table2
    on table1.x = table2.x
    """,
    ],
)
@assert_state_not_change
def test_joining_two_subqueries_with_overlapping_columns_same_table(sql, forest_fires):
    my_table = query(sql)
    columns = ["X", "Y", "rain"]

    def get_select_rename_columns(alias: str):
        my_columns = forest_fires.get_columns(columns)
        renamed_columns = []
        for i, column in enumerate(my_columns):
            renamed_columns.append(column.name(f"{alias}.{columns[i]}"))
        return my_columns, renamed_columns

    select1, renamed1 = get_select_rename_columns("table1")
    select2, renamed2 = get_select_rename_columns("table2")
    subquery1 = forest_fires[select1]
    subquery2 = forest_fires[select2]
    joined = subquery1.join(
        subquery2, predicates=subquery1.X == subquery2.X
    ).projection(renamed1 + renamed2)
    assert_ibis_equal_show_diff(joined, my_table)


@pytest.mark.parametrize(
    "sql",
    [
        """
    select * from
    ((select type, attribute, power from digimon_move_list) table1
    join
    (select type, attribute, digimon from
    digimon_mon_list) table2
    on table1.type = table2.type) sub
    """,
        """
    select * from
    ((select type, attribute, power from digimon_move_list) table1
    join
    (select type, attribute, digimon from digimon_mon_list) table2
    on table1.type = table2.type) sub
    """,
    ],
)
@assert_state_not_change
def test_joining_two_subqueries_with_overlapping_columns_different_tables(
    sql, digimon_mon_list, digimon_move_list
):
    my_table = query(sql)
    subquery1 = digimon_move_list[
        [
            digimon_move_list.Type.name("type"),
            digimon_move_list.Attribute.name("attribute"),
            digimon_move_list.Power.name("power"),
        ]
    ]
    subquery2 = digimon_mon_list[
        [
            digimon_mon_list.Type.name("type"),
            digimon_mon_list.Attribute.name("attribute"),
            digimon_mon_list.Digimon.name("digimon"),
        ]
    ]
    ibis_table = subquery1.join(
        subquery2, predicates=subquery1.type == subquery2.type
    ).projection(
        [
            subquery1.type.name("table1.type"),
            subquery1.attribute.name("table1.attribute"),
            subquery1.power.name("power"),
            subquery2.type.name("table2.type"),
            subquery2.attribute.name("table2.attribute"),
            subquery2.digimon,
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_multi_column_joins(time_data):
    my_table = query(
        """
    SELECT
        table1.team,
        table1.start_time_count,
        table2.start_time_count_d
    FROM
    (SELECT
        team,
        count(start_time)
        AS start_time_count
    FROM
        time_data
        GROUP BY team) table1
        INNER JOIN
    (SELECT team, count(start_time) AS start_time_count_d FROM
        (SELECT distinct team, start_time FROM time_data) intermediate GROUP BY team
    ) table2
        ON
            table1.team = table2.team AND
            table1.start_time_count = table2.start_time_count_d
    """
    )
    table1 = time_data.group_by(time_data.team).aggregate(
        [time_data.start_time.count().name("start_time_count")]
    )
    intermediate = time_data.projection(
        [time_data.team, time_data.start_time]
    ).distinct()
    table2 = intermediate.group_by(intermediate.team).aggregate(
        [intermediate.start_time.count().name("start_time_count_d")]
    )
    ibis_table = table1.join(
        table2,
        predicates=(
            (table1.team == table2.team)
            & (table1.start_time_count == table2.start_time_count_d)
        ),
        how="inner",
    ).projection([table1.team, table1.start_time_count, table2.start_time_count_d])
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_column_values_in_subquery(digimon_move_list):
    my_table = query(
        """
    select move, type, power from
    digimon_move_list
    where
    power in
        ( select max(power) as power
         from digimon_move_list
         group by type ) t1
    """
    )
    subquery = (
        digimon_move_list.groupby(digimon_move_list.get_column("Type").name("type"))
        .aggregate(digimon_move_list.Power.max().name("power"))
        .drop(["type"])
    )
    ibis_table = digimon_move_list.filter(
        digimon_move_list.Power.isin(subquery.get_column("power"))
    ).projection(
        [
            digimon_move_list.Move.name("move"),
            digimon_move_list.Type.name("type"),
            digimon_move_list.Power.name("power"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_column_values_in_other_table(digimon_move_list, digimon_mon_list):
    my_table = query(
        """
    select power from
    digimon_move_list
    where
    type in
        ( select type
         from digimon_mon_list ) t1
    """
    )
    digimon_mon_list_lower_name = digimon_mon_list.projection(
        [digimon_mon_list.Type.name("type")]
    )
    ibis_table = digimon_move_list.filter(
        digimon_move_list.Type.isin(digimon_mon_list_lower_name.type)
    ).projection(
        [
            digimon_move_list.Power.name("power"),
        ]
    )
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_limit(forest_fires):
    """
    Test limit clause
    :return:
    """
    my_table = query("""select * from forest_fires limit 10""")
    ibis_table = forest_fires.head(10)
    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
def test_join_with_overlapping_column_names(digimon_mon_list, digimon_move_list):
    query_text = """
    SELECT mon_list.attribute as attribute
    FROM digimon_mon_list as mon_list
    inner join digimon_move_list as move_list
    on move_list.attribute=mon_list.attribute
    where memory < 70
    """
    my_table = query(query_text)
    renamed_columns = resolved_columns(
        digimon_mon_list, digimon_move_list, "mon_list", "move_list"
    )
    joined_tables = digimon_mon_list.join(
        digimon_move_list,
        predicates=digimon_move_list.Attribute == digimon_mon_list.Attribute,
        how="inner",
    )[renamed_columns]
    filtered = joined_tables.filter(digimon_mon_list.Memory < 70)
    ibis_table = filtered[[digimon_mon_list.Attribute.name("attribute")]]
    assert_ibis_equal_show_diff(ibis_table, my_table)


def get_columns(table: TableExpr):
    return table.get_columns(table.columns)


@join_params
@assert_state_not_change
def test_join_more_than_2_tables(
    multitable_join_main_table,
    multitable_join_lookup_table,
    multitable_join_relationship_table,
    multitable_join_promotion_table,
    sql_join: str,
    ibis_join: str,
):
    query_text = f"""
    SELECT multi_main.id
    FROM multi_main
    {sql_join} join multi_lookup
    on multi_main.lookup_id = multi_lookup.id
    {sql_join} join multi_relationship
    on multi_main.relationship_id = multi_relationship.id
    {sql_join} join multi_promotion
    on multi_main.promotion_id = multi_promotion.id
    """
    my_table = query(query_text)
    join_type = ibis_join
    ibis_table = (
        multitable_join_main_table.join(
            multitable_join_lookup_table,
            predicates=multitable_join_main_table.lookup_id
            == multitable_join_lookup_table.id,
            how=join_type,
        )
        .join(
            multitable_join_relationship_table,
            predicates=multitable_join_main_table.relationship_id
            == multitable_join_relationship_table.id,
            how=join_type,
        )
        .join(
            multitable_join_promotion_table,
            predicates=multitable_join_main_table.promotion_id
            == multitable_join_promotion_table.id,
            how=join_type,
        )
        .projection([multitable_join_main_table.id])
    )

    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
@pytest.mark.skip("Need to implement this")
def test_cross_join_more_than_2_tables(
    multitable_join_main_table,
    multitable_join_lookup_table,
    multitable_join_relationship_table,
    multitable_join_promotion_table,
):
    query_text = """
    SELECT multi_main.id
    FROM multi_main
    cross join multi_lookup
    cross join multi_relationship
    cross join multi_promotion
    """
    my_table = query(query_text)
    ibis_table = (
        multitable_join_main_table.cross_join(
            multitable_join_lookup_table,
        )
        .cross_join(
            multitable_join_relationship_table,
        )
        .cross_join(
            multitable_join_promotion_table,
        )
        .projection([multitable_join_main_table.id])
    )

    assert_ibis_equal_show_diff(ibis_table, my_table)


@assert_state_not_change
@join_params
def test_join_without_overlapping_columns(
    multitable_join_main_table,
    multitable_join_promotion_table_no_overlap,
    sql_join: str,
    ibis_join: str,
):
    my_table = query(
        f"""
    select id, promotion from multi_main {sql_join} join
    multi_promotion_no_overlap
    on id = other_id
    """
    )
    join_type = ibis_join
    ibis_table = multitable_join_main_table.join(
        multitable_join_promotion_table_no_overlap,
        predicates=multitable_join_main_table.id
        == multitable_join_promotion_table_no_overlap.other_id,
        how=join_type,
    )[
        multitable_join_main_table.id,
        multitable_join_promotion_table_no_overlap.promotion,
    ]
    assert_ibis_equal_show_diff(ibis_table, my_table)


# @assert_state_not_change
# def test_join_with_alias():
#     query_text = """
#     SELECT mon_list.attribute as attribute
#     FROM digimon_mon_list
#     left join digimon_move_list
#     on move_list.attribute=mon_list.attribute
#     left join mon_list
#     on mon_list.number=move_list.sp_cost
#     """
#     my_table = query(query_text)
