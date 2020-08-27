from sql_to_ibis.sql.sql_value_objects import Bool, Date, Number, String

GIVEN_TYPE_TO_IBIS = {
    "object": "varchar",
    "datetime64": "timestamp",
    "datetime": "timestamp",
    "smallint": "int16",
    "int": "int32",
    "bigint": "int64",
}

TYPE_TO_SQL_TYPE = {
    "object": String,
    "string": String,
    "int64": Number,
    "float64": Number,
    "bool": Bool,
    "datetime64": Date,
    "date": Date,
}


def to_ibis_type(given_type: str):
    """
    Returns the corresponding ibis dtype
    :return:
    """
    if given_type in GIVEN_TYPE_TO_IBIS:
        return GIVEN_TYPE_TO_IBIS[given_type]
    return given_type
