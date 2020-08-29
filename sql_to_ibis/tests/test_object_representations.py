from sql_to_ibis.sql.sql_value_objects import Value, DerivedColumn, JoinBase, Table

def test_value_repr(time_data):
    value = Value(time_data.team, alias="my_team", typename="string")
    assert repr(value) == "Value(final_name=my_team, value=IbisStringColumn(), " \
                          "alias=my_team, type=string"

def test_derived_column(time_data):
    column = DerivedColumn(time_data.team, typename="string",
                           function="sum")

    assert repr(column) == """DerivedColumn(final_name=ref_0
PandasTable[table]
  name: TIME_DATA
  schema:
    duration_seconds : int64
    start_time : string
    end_time : string
    count : int64
    person : string
    team : string

team = Column[string*] 'team' from table
  ref_0, value=IbisStringColumn(), type=string, function=sum)"""
