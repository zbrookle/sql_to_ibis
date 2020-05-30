"""
Module containing all lark internal_transformer classes
"""
from datetime import date, datetime
import re
from types import FunctionType
from typing import Dict, List, Tuple, Union

from lark import Token, Transformer, Tree, v_args
from pandas import Series, concat, merge
from ibis.expr.types import TableExpr
from ibis.expr.operations import TableColumn
from ibis.expr.api import NumericColumn
import ibis

from sql_to_ibis.exceptions.sql_exception import TableExprDoesNotExist
from sql_to_ibis.sql_objects import (
    Aggregate,
    AmbiguousColumn,
    Bool,
    Column,
    Date,
    DerivedColumn,
    Expression,
    Join,
    Literal,
    Number,
    QueryInfo,
    String,
    Subquery,
    Value,
    ValueWithPlan,
)

ORDER_TYPES = ["asc", "desc", "ascending", "descending"]
ORDER_TYPES_MAPPING = {
    "asc": "asc",
    "desc": "desc",
    "ascending": "asc",
    "descending": "desc",
}
GET_TABLE_REGEX = re.compile(
    r"^(?P<table>[a-z_]\w*)\.(?P<column>[a-z_]\w*)$", re.IGNORECASE
)
PANDAS_TYPE_PYTHON_TYPE_FUNCTION = {
    "object": str,
    "string": str,
    "int16": int,
    "int32": int,
    "int64": int,
    "float16": float,
    "float32": float,
    "float64": float,
    "bool": bool,
}

TYPE_TO_PANDAS_TYPE = {
    "varchar": "string",
    "smallint": "int16",
    "int": "int32",
    "bigint": "int64",
    "float": "float64",
    "timestamp": "datetime64",
    "datetime64": "datetime64",
    "timedelta[ns]": "timedelta[ns]",
    "category": "category",
}

for TYPE in PANDAS_TYPE_PYTHON_TYPE_FUNCTION:
    TYPE_TO_PANDAS_TYPE[TYPE] = TYPE

TYPE_TO_SQL_TYPE = {
    "object": String,
    "string": String,
    "int64": Number,
    "float64": Number,
    "bool": Bool,
    "datetime64": Date,
}

GIVEN_TYPE_TO_IBIS = {"object": "varchar"}

from sql_to_ibis.parsing.aggregation_aliases import AVG_AGGREGATIONS, \
    MAX_AGGREGATIONS, MIN_AGGREGATIONS, NUMERIC_AGGREGATIONS, SUM_AGGREGATIONS


def num_eval(arg):
    """
    Takes an argument that may be a string or number and outputs a number
    :param arg:
    :return:
    """
    assert isinstance(arg, (Token, float, int))
    if isinstance(arg, str):
        # pylint: disable=eval-used
        return eval(arg)
    return arg


def get_wrapper_value(value):
    """
    If the value is a literal return it's value
    :param value:
    :return:
    """
    if isinstance(value, Value):
        return value.get_value()
    return value

def to_ibis_type(given_type: str):
    """
    Returns the corresponding ibis dtype
    :return:
    """
    if given_type in GIVEN_TYPE_TO_IBIS:
        return GIVEN_TYPE_TO_IBIS[given_type]
    return given_type


class TransformerBaseClass(Transformer):
    """
    Base class for transformers
    """

    def __init__(
        self,
        dataframe_name_map=None,
        dataframe_map=None,
        column_name_map=None,
        column_to_dataframe_name=None,
        _temp_dataframes_dict=None,
    ):
        Transformer.__init__(self, visit_tokens=False)
        self.dataframe_name_map = dataframe_name_map
        self.dataframe_map = dataframe_map
        self.column_name_map = column_name_map
        self.column_to_dataframe_name = column_to_dataframe_name
        self._temp_dataframes_dict = _temp_dataframes_dict
        self._execution_plan = ""

    def get_table(self, frame_name) -> TableExpr:
        """
        Returns the dataframe with the name given
        :param frame_name:
        :return:
        """
        if isinstance(frame_name, Token):
            frame_name = frame_name.value
        if isinstance(frame_name, Subquery):
            frame_name = frame_name.name
        if isinstance(frame_name, Join):
            return frame_name
        return self.dataframe_map[frame_name]

    def set_column_value(self, column: Column) -> None:
        """
        Sets the column value based on what it is in the dataframe
        :param column:
        :return:
        """
        if column.name != "*":
            dataframe_name = self.column_to_dataframe_name[column.name.lower()]
            if isinstance(dataframe_name, AmbiguousColumn):
                raise Exception(f"Ambiguous column reference: {column.name}")
            dataframe = self.get_table(dataframe_name)
            column_true_name = self.column_name_map[dataframe_name][column.name.lower()]
            column.value = dataframe[column_true_name]
            column.table = dataframe_name

    def column_name(self, name_list_format: List[str]):
        """
        Returns a column token_or_tree with the name extracted
        :param name_list_format: List formatted name
        :return: Tree with column token_or_tree
        """
        name = "".join(name_list_format)
        column = Column(name="".join(name))
        self.set_column_value(column)
        return column


def boolean_decorator(boolean_operator: str):
    """
    Returns a function that wraps around the given boolean function
    :param boolean_operator:
    :return:
    """

    def boolean_function(function: FunctionType):
        def wrapper(self, expressions: list):
            plan = self.create_execution_plan_expression(*expressions, boolean_operator)
            result = function(self, expressions)
            return ValueWithPlan(result, plan)

        return wrapper

    return boolean_function


class InternalTransformer(TransformerBaseClass):
    """
    Evaluates subtrees with knowledge of provided tables that are in the proper scope
    """

    def __init__(
        self, tables, dataframe_map, column_name_map, column_to_dataframe_name
    ):
        TransformerBaseClass.__init__(
            self, dataframe_map=dataframe_map, column_name_map=column_name_map
        )
        self.tables = [
            table.name if isinstance(table, Subquery) else table for table in tables
        ]
        self.column_to_dataframe_name = {}
        for column in column_to_dataframe_name:
            table = column_to_dataframe_name.get(column)
            if isinstance(table, AmbiguousColumn):
                table_name = self.tables[0]
                if table_name in table.tables:
                    self.column_to_dataframe_name[column] = table_name
            if table in self.tables:
                self.column_to_dataframe_name[column] = table

        # These variables need instance scope for rank instance method
        self.partition_func_dict = {
            self.set_rank_regular: self.set_rank_regular_partition,
            self.set_rank_dense: self.set_rank_dense_partition,
        }
        self.partition_rank_counter: Dict[str, int] = {}
        self.partition_rank_offset: Dict[str, int] = {}
        self.rank_counter = 1
        self.rank_offset = 0
        self.rank_map = {}
        self.last_key = None

    def transform(self, tree):
        new_tree = TransformerBaseClass.transform(self, tree)
        if isinstance(new_tree, Token) and isinstance(new_tree.value, ValueWithPlan):
            new_tree.value = new_tree.value.value
        return new_tree

    def mul(self, args: Tuple[int, int]):
        """
        Returns the product two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) * num_eval(arg2)

    def expression_mul(self, args: Tuple):
        """
        Returns the product of two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 * arg2

    def add(self, args: Tuple):
        """
        Returns the sum two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) + num_eval(arg2)

    def expression_add(self, args: Tuple):
        """
        Returns the sum of two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 + arg2

    def sub(self, args: Tuple):
        """
        Returns the difference between two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) - num_eval(arg2)

    def expression_sub(self, args: Tuple):
        """
        Returns the difference between two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 - arg2

    def div(self, args: Tuple):
        """
        Returns the division of two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) / num_eval(arg2)

    def expression_div(self, args):
        """
        Returns the difference between two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 / arg2

    def number(self, numerical_value):
        """
        Return a number token_or_tree with a numeric value as a child
        :param numerical_value:
        :return:
        """
        return Number(num_eval(numerical_value[0]))

    def string(self, string_token):
        """
        Return value of the token_or_tree associated with the string
        :param string_token:
        :return:
        """
        return String(string_token[0].value)

    def timestamp_expression(self, date_list):
        """
        Return a timestamp object
        :param date_list:
        :return:
        """
        return date_list[0]

    @staticmethod
    def int_token_list(token_list):
        """
        Returns a list of integer from a list of tokens
        :param token_list:
        :return:
        """
        return [int(token.value) for token in token_list]

    def date(self, date_list):
        """
        Returns list with correct date integers
        :param date_list:
        :return:
        """
        return self.int_token_list(date_list)

    def time(self, time_list):
        """
        Returns list with correct time integers
        :param time_list:
        :return:
        """
        return self.int_token_list(time_list)

    def custom_timestamp(self, datetime_list):
        """
        Return a custom time stamp based on user input
        :param datetime_list:
        :return:
        """
        return Date(datetime(*(datetime_list[0] + datetime_list[1])))

    def datetime_now(self, *extra_args):
        """
        Return current date and time
        :param extra_args: Arguments that lark parser must pass in
        :return:
        """
        date_value = Date(datetime.now())
        date_value.set_alias("now()")
        return date_value

    def date_today(self, *extra_args):
        """
        Return current date
        :param extra_args: Arguments that lark parser must pass in
        :return:
        """
        date_value = Date(date.today())
        date_value.set_alias("today()")
        return date_value

    def create_execution_plan_expression(
        self, expression1: Value, expression2: Value, relationship
    ):
        """
        Returns the execution plan for both expressions taking relationship into account

        :param expression1:
        :param expression2:
        :param relationship:
        :return:
        """
        return (
            f"{expression1.get_plan_representation()}{relationship}"
            f"{expression2.get_plan_representation()}"
        )

    @boolean_decorator("==")
    def equals(self, expressions):
        """
        Compares two expressions for equality
        :param expressions:
        :return:
        """
        return expressions[0] == expressions[1]

    def not_equals(self, expressions):
        """
        Compares two expressions for inequality
        :param expressions:
        :return:
        """
        plan_expr = self.create_execution_plan_expression(
            expressions[0], expressions[1], "=="
        )
        return ValueWithPlan(~(expressions[0] == expressions[1]), f"~({plan_expr})")

    @boolean_decorator(">")
    def greater_than(self, expressions):
        """
        Performs a greater than sql_object
        :param expressions:
        :return:
        """
        return expressions[0] > expressions[1]

    @boolean_decorator(">=")
    def greater_than_or_equal(self, expressions):
        """
        Performs a greater than or equal sql_object
        :param expressions:
        :return:
        """
        return expressions[0] >= expressions[1]

    @boolean_decorator("<")
    def less_than(self, expressions):
        """
        Performs a less than sql_object
        :param expressions:
        :return:
        """
        return expressions[0] < expressions[1]

    @boolean_decorator("<=")
    def less_than_or_equal(self, expressions):
        """
        Performs a less than or equal sql_object
        :param expressions:
        :return:
        """
        return expressions[0] <= expressions[1]

    def between(self, expressions: List[Value]):
        """
        Performs a less than or equal and greater than or equal
        :param expressions:
        :return:
        """
        main_expression = expressions[0]
        between_expressions = expressions[1:]
        plan = main_expression.get_plan_representation()
        plan += (
            f".between({between_expressions[0].get_plan_representation()}, "
            f"{between_expressions[1].get_plan_representation()})"
        )

        return ValueWithPlan(main_expression.value.between(*between_expressions), plan)

    def in_expr(self, expressions: List[Value]):
        """
        Evaluate in sql_object
        :param expressions:
        :return:
        """
        in_list = [expression.get_value() for expression in expressions[1:]]
        plan = expressions[0].get_plan_representation()
        plan += f".isin({in_list})"
        return ValueWithPlan(expressions[0].value.isin(in_list), plan)

    def not_in_expr(self, expressions: List[Value]):
        """
        Negate in expr
        :param expressions:
        :return:
        """
        in_value = self.in_expr(expressions)

        return ValueWithPlan(~in_value.value, "~" + in_value.execution_plan)

    def bool_expression(self, expression: List[ValueWithPlan]) -> ValueWithPlan:
        """
        Return the bool sql_object
        :param expression:
        :return: boolean sql_object
        """
        return expression[0]

    def bool_and(self, truth_series_pair: List[Value]) -> ValueWithPlan:
        """
        Return the truth value of the series pair
        :param truth_series_pair:
        :return:
        """
        plans: List[str] = []
        truth_series_pair_values: List[Series] = []
        for i, value in enumerate(truth_series_pair):
            truth_series_pair_values.append(value.get_value())
            plans.append(value.get_plan_representation())

        return ValueWithPlan(
            truth_series_pair_values[0] & truth_series_pair_values[1],
            f"{plans[0]} & {plans[1]}",
        )

    def bool_parentheses(self, bool_expression_in_list: list):
        return bool_expression_in_list[0]

    def bool_or(self, truth_series_pair):
        """
        Return the truth value of the series pair
        :param truth_series_pair:
        :return:
        """
        return truth_series_pair[0] | truth_series_pair[1]

    def comparison_type(self, comparison):
        """
        Return the comparison

        :param comparison:
        :return:
        """
        return comparison[0]

    def negated_bool_expression(self, bool_expression):
        """
        Returns a negated boolean sql_object
        :param bool_expression:
        :return:
        """
        print("Negated bool", bool_expression)

    def where_expr(self, truth_value_dataframe):
        """
        Return a where token_or_tree
        :param truth_value_dataframe:
        :return: Token
        """
        return Token("where_expr", truth_value_dataframe[0])

    def alias_string(self, name: List[str]):
        """
        Returns an alias token_or_tree with the name extracted
        :param name:
        :return:
        """
        return Tree("alias", str(name[0]))

    def from_expression(self, expression):
        """
        Return a from sql_object token_or_tree
        :param expression:
        :return: Token from sql_object
        """
        expression = expression[0]
        if isinstance(expression, (Subquery, Join)):
            value = expression
        else:
            value = expression.value
        return Token("from_expression", value)

    def when_then(self, when_then_values):
        """
        When / then sql_object
        :param when_then_values:
        :return:
        """
        return when_then_values[0], when_then_values[1]

    def case_expression(
        self, when_expressions: List[Union[Tuple[Value, Value], Value]]
    ):
        """
        Handles sql_to_ibis case expressions
        :param when_expressions:
        :return:
        """
        case_execution_plan = "NONE_SERIES"
        if isinstance(when_expressions[0], tuple):
            dataframe_size = when_expressions[0][0].value.size
        new_column = Series(data=[None for _ in range(0, dataframe_size)])
        current_truth_value = Series(data=[False for _ in range(0, dataframe_size)])

        current_truth_value_plan = (
            "FALSE_SERIES"  # See the README section why this is here
        )

        for i, when_expression in enumerate(when_expressions):
            if isinstance(when_expression, tuple):
                conditional_object = when_expression[0]
                expression_truth_value = conditional_object.get_value()
                new_column = new_column.mask(
                    (expression_truth_value ^ current_truth_value)
                    & expression_truth_value,
                    when_expression[1].get_value(),
                )
                expression_truth_value_plan = (
                    conditional_object.get_plan_representation()
                )
                case_execution_plan += (
                    f".mask((({expression_truth_value_plan}) ^ "
                    f"({current_truth_value_plan})) & ({expression_truth_value_plan}), "
                    f"{when_expression[1].get_plan_representation()})"
                )
                current_truth_value = current_truth_value | expression_truth_value
                current_truth_value_plan = (
                    f"({current_truth_value_plan}) | "
                    f"({expression_truth_value_plan})"
                )
            else:
                conditional_object = when_expression
                new_column = new_column.where(
                    current_truth_value, conditional_object.get_value()
                )
                case_execution_plan += (
                    f".where({current_truth_value_plan}, "
                    f"{conditional_object.get_plan_representation()})"
                )
        return Expression(value=new_column, execution_plan=case_execution_plan)

    def rank_form(self, form):
        """
        Returns the rank form
        :param form:
        :return:
        """
        return form

    def order_asc(self, column):
        """
        Return sql_object in asc order
        :param column:
        :return:
        """
        column = column[0]
        column.value = column.value.copy().sort_values(ascending=True)
        return Token("order", (column, True))

    def order_desc(self, column):
        """
        Return sql_object in asc order
        :param column:
        :return:
        """
        column = column[0]
        column.value = column.value.copy().sort_values(ascending=False)
        return Token("order", (column, False))

    def set_rank_dense(self, row):
        """
        Set rank dense rank without gaps between consecutive ranks
        :param row: A pandas row object
        :return: The dense rank of the current row
        """
        key = str(list(row))
        if self.rank_map.get(key):
            return self.rank_map[key]
        self.rank_map[key] = self.rank_counter
        rank = self.rank_counter
        self.rank_counter += 1
        return rank

    def set_rank_regular(self, row):
        """
        Set rank in traditional SQL database fashion
        :param row: A pandas row object
        :return: The rank of the current row
        """
        key = str(list(row))
        if self.rank_map.get(key):
            self.rank_offset += 1
            return self.rank_map[key]
        rank = self.rank_counter + self.rank_offset
        self.rank_map[key] = rank
        self.rank_counter += 1
        return rank

    def set_rank_regular_partition(self, row, partition_slice):
        """
        Set rank with a partition
        :param row: A pandas row object
        :param partition_slice: Integer to slice the values in the row based on
        partition
        :return: The partitioned rank of the current row
        """
        row_list = list(row)[:partition_slice]
        partition_list = list(row)[partition_slice:]
        key = str(row_list)
        partition_key = str(partition_list)
        if self.rank_map.get(partition_key):
            if self.rank_map[partition_key].get(key):
                self.partition_rank_offset[partition_key] += 1
                return self.rank_map[partition_key][key]
            self.partition_rank_counter[partition_key] += 1
            rank = (
                self.partition_rank_counter[partition_key]
                + self.partition_rank_offset[partition_key]
            )
            self.rank_map[partition_key][key] = rank
        else:
            rank = 1
            self.rank_map[partition_key] = {}
            self.partition_rank_counter[partition_key] = rank
            self.partition_rank_offset[partition_key] = 0
            self.rank_map[partition_key][key] = rank
        return rank

    def set_rank_dense_partition(self, row, partition_slice):
        """
        Set rank with a partition
        :param row: A pandas row object
        :param partition_slice: Integer to slice the values in the row based on
        partition
        :return: The partitioned rank of the current row
        """
        row_list = list(row)[:partition_slice]
        partition_list = list(row)[partition_slice:]
        key = str(row_list)
        partition_key = str(partition_list)
        if self.rank_map.get(partition_key):
            if self.rank_map[partition_key].get(key):
                return self.rank_map[partition_key][key]
            self.partition_rank_counter[partition_key] += 1
            rank = self.partition_rank_counter[partition_key]
            self.rank_map[partition_key][key] = rank
        else:
            rank = 1
            self.rank_map[partition_key] = {}
            self.partition_rank_counter[partition_key] = 1
            self.rank_map[partition_key][key] = rank
        return rank

    def partition_by(self, column_list):
        """
        Returns a partition token_or_tree containing the corresponding column
        :param column_list: List containing only one column
        :return:
        """
        column = column_list[0]
        return Token("partition", column)

    def rank(self, tokens, rank_function):
        """
        Returns the evaluated rank expressions
        :param tokens: Tokens making up the rank sql_object
        :param rank_function: Function to be used in rank evaluation
        :return:
        """
        expressions = tokens[0]
        series_list = []
        order_list = []
        partition_list = []
        for token in expressions:
            if token.type == "order":
                column = token.value[0]
                ascending = token.value[1]
                series_list.append(column.value)
                order_list.append(ascending)
            elif token.type == "partition":
                column = token.value
                partition_list.append(column.value)
        rank_df = concat(series_list, axis=1)
        column_names = rank_df.columns.to_list()
        if partition_list:
            rank_df = concat(series_list + partition_list, axis=1)
            rank_function = self.partition_func_dict[rank_function]
            column_slice = len(column_names)
        rank_df.sort_values(by=column_names, ascending=order_list, inplace=True)
        if partition_list:
            rank = rank_df.apply(rank_function, args=(column_slice,), axis=1)
            self.partition_rank_counter = {}
            self.partition_rank_offset = {}
        else:
            rank = rank_df.apply(rank_function, axis=1)
            self.rank_counter = 1
            self.rank_offset = 0
        self.rank_map = {}
        rank_df["rank"] = rank
        return Expression(value=rank)

    def rank_expression(self, tokens):
        """
        Handles rank expressions
        :param tokens:
        :return:
        """
        return self.rank(tokens, self.set_rank_regular)

    def dense_rank_expression(self, tokens):
        """
        Handles dense_rank_expressions
        :param tokens:
        :return:
        """
        return self.rank(tokens, self.set_rank_dense)

    def select_expression(self, expression_and_alias):
        """
        Returns the appropriate object for the given sql_object
        :param expression_and_alias: An sql_object token_or_tree and
              A token_or_tree containing the name to be assigned
        :return:
        """
        expression = expression_and_alias[0]
        alias = None
        if len(expression_and_alias) == 2:
            alias = expression_and_alias[1]
        if isinstance(expression, Tree):
            value = expression.children
            if expression.data == "sql_aggregation":
                function = value[0]
                value = value[1]
                expression = Aggregate(value=value, function=function)

        if alias:
            expression.set_alias(alias.children)
        return expression

    def join(self, *args):
        """
        Extracts the join sql_object
        :param args: Arguments that are passed to the join
        :return: join sql_object
        """
        return args[0]

    def group_by(self, column):
        """
        Returns a group token_or_tree
        :param column: Column to group by
        :return: group token_or_tree
        """
        column = column[0]
        return Token("group", str(column.name))

    def as_type(self, column_and_type):
        """
        Extracts token_or_tree type and returns tree object with sql_object and type
        :param sql_object: Expression to be evaluated / the name of a column
        :param typename: Data type
        :return:
        """
        column = column_and_type[0]
        typename = column_and_type[1]
        column.typename = TYPE_TO_PANDAS_TYPE[typename.value]
        return column

    def literal_cast(self, value_and_type: list):
        """
        Cast variable as the given given_type for a literal
        :param value_and_type: Value and pandas dtype to be cast as
        :return:
        """
        value_wrapper, given_type = value_and_type
        if given_type == "datetime64":
            date_value = datetime.strptime(value_wrapper.value, "%Y-%m-%d")
            return Date(date_value)
        new_type = TYPE_TO_SQL_TYPE[given_type]
        new_value = new_type(value_wrapper.value.cast(to_ibis_type(given_type)))
        return new_value


# pylint: disable=no-self-use
class HavingTransformer(TransformerBaseClass):
    """
    Transformer for having clauses since group by needs to be applied first
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, tables, group_by, dataframe_map, column_name_map, column_to_dataframe_name
    ):
        self.tables = tables
        self.group_by = group_by
        TransformerBaseClass.__init__(
            self,
            dataframe_map=dataframe_map,
            column_name_map=column_name_map,
            column_to_dataframe_name=column_to_dataframe_name,
        )

    def transform(self, tree: Tree):
        new_tree, plan = TransformerBaseClass.transform(self, tree)
        new_tree.children[0] = new_tree.children[0].value
        return new_tree, plan

    def aggregate(self, function_name_list_form):
        """
        Return the string representation fo aggregate function name instead of list
        :param function_name_list_form:
        :return:
        """
        return "".join(function_name_list_form)

    def sql_aggregation(self, aggregation_expr: list):
        """
        Handles presence of aggregation in an sql_object
        :param aggregation_expr: Function sql_object
        :return:
        """
        aggregate_token: Token = aggregation_expr[0]
        aggregate_name: str = aggregate_token.value
        column = aggregation_expr[1]
        table = self.dataframe_map[column.table]
        column_true_name = self.column_name_map[column.table][column.name]
        aggregates = {column_true_name: aggregate_name}
        if self.group_by:
            new_series = (
                table.groupby(self.group_by).aggregate(aggregates).reset_index()
            )
            aggregation_plan = (
                f"{column.table}.groupby({self.group_by}).aggregate("
                f"{aggregates}).reset_index()"
            )
        else:
            new_series = table.aggregate(aggregates).to_frame().transpose()
            aggregation_plan = (
                f"{column.table}.aggregate({aggregates}).to_frame()" f".transpose()"
            )
        aggregation_plan += f"[{column_true_name}]"
        return ValueWithPlan(new_series[column_true_name], aggregation_plan)

    def having_expr(self, having_expr):
        """
        Handles having expressions
        :param having_expr:
        :return:
        """
        internal_transformer = InternalTransformer(
            self.tables,
            self.dataframe_map,
            self.column_name_map,
            self.column_to_dataframe_name,
        )
        having_expr = Tree("having_expr", having_expr)
        return internal_transformer.transform(having_expr)


# pylint: disable=no-self-use, super-init-not-called
@v_args(inline=True)
class SQLTransformer(TransformerBaseClass):
    """
    Transformer for the lark sql_to_ibis parser
    """

    def __init__(
        self,
        dataframe_name_map=None,
        dataframe_map=None,
        column_name_map=None,
        column_to_dataframe_name=None,
    ):
        if dataframe_name_map is None:
            dataframe_name_map = {}
        if dataframe_map is None:
            dataframe_map = {}
        if column_name_map is None:
            column_name_map = {}
        if column_to_dataframe_name is None:
            column_to_dataframe_name = {}
        TransformerBaseClass.__init__(
            self,
            dataframe_name_map,
            dataframe_map,
            column_name_map,
            column_to_dataframe_name,
            _temp_dataframes_dict={},
        )

    def add_column_to_column_to_dataframe_name_map(self, column, table):
        """
        Adds a column to the column_to_dataframe_name_map
        :param column:
        :param table:
        :return:
        """
        if self.column_to_dataframe_name.get(column) is None:
            self.column_to_dataframe_name[column] = table
        elif isinstance(self.column_to_dataframe_name[column], AmbiguousColumn):
            self.column_to_dataframe_name[column].tables.append(table)
        else:
            original_table = self.column_to_dataframe_name[column]
            self.column_to_dataframe_name[column] = AmbiguousColumn(
                [original_table, table]
            )

    def table(self, table_name, alias=""):
        """
        Check for existence of pandas dataframe with same name
        If not exists raise TableExprDoesNotExist
        Otherwise return the name of the actual TableExpr
        :return:
        """
        table_name = table_name.lower()
        if table_name not in self.dataframe_name_map:
            raise TableExprDoesNotExist(table_name)
        if alias:
            self.dataframe_name_map[alias] = self.dataframe_name_map[table_name]
        return Token("table", self.dataframe_name_map[table_name])

    def order_by_expression(self, rank_tree):
        """
        Returns the column name for the order sql_object
        :param rank_tree: Tree containing order info
        :return:
        """
        order_type = rank_tree.data
        ascending = order_type == "order_asc"
        return Token("order_by", (rank_tree.children[0].children, ascending))

    def integer(self, integer_token):
        """
        Returns the integer value
        :param integer_token:
        :return:
        """
        integer_value = int(integer_token.value)
        return integer_value

    def limit_count(self, limit_count_value):
        """
        Returns a limit token_or_tree
        :param limit_count_value:
        :return:
        """
        return Token("limit", limit_count_value)

    def query_expr(self, query_info: QueryInfo, *args):
        """
        Handles the full query, including order and set operations such as union
        :param query_info: Map of all query information
        :param args: Additional arguments aside from query info
        :return: Query info
        """
        for token in args:
            if isinstance(token, Token):
                if token.type == "order_by":
                    query_info.order_by.append(token.value)
                elif token.type == "limit":
                    query_info.limit = token.value
        return query_info

    def subquery(self, query_info, alias):
        """
        Handle subqueries amd return a subquery object
        :param query_info:
        :param alias:
        :return:
        """
        alias_name = alias.children[0].value
        self.dataframe_map[alias_name] = self.to_ibis_table(query_info)
        subquery = Subquery(name=alias_name, query_info=query_info)
        self.column_name_map[alias_name] = {}
        for column in self.dataframe_map[alias_name].columns:
            self.add_column_to_column_to_dataframe_name_map(column.lower(), alias_name)
            self.column_name_map[alias_name][column.lower()] = column
        return subquery

    def column_name(self, *names):
        full_name = ".".join([str(name) for name in names])
        return Tree("column_name", full_name)

    def join(self, join_expression):
        """
        Handle join tree
        :param join_expression:
        :return:
        """
        return join_expression

    def get_lower_columns(self, table_name):
        """
        Returns a list of lower case column names for a given table name
        :param column_list:
        :return:
        """
        return [column.lower() for column in list(self.get_table(table_name).columns)]

    def determine_column_side(self, column, left_table, right_table):
        """
        Check if column table prefix is one of the two tables (if there is one) AND
        the column has to be in one of the two tables
        """
        column_match = GET_TABLE_REGEX.match(column)
        column_table = ""
        if column_match:
            column = column_match.group("column").lower()
            column_table = column_match.group("table").lower()

        left_columns = self.column_name_map[left_table]
        right_columns = self.column_name_map[right_table]
        if column not in left_columns and column not in right_columns:
            raise Exception("Column not found")

        left_table = left_table.lower()
        right_table = right_table.lower()
        if column_table:
            if column_table == left_table and column in left_columns:
                return "left", column
            if column_table == right_table and column in right_columns:
                return "right", column
            raise Exception("Table specified in join columns not present in join")
        if column in left_columns and column in right_columns:
            raise Exception(
                f"Ambiguous column: {column}\nSpecify table name with table_name"
                f".{column}"
            )
        if column in left_columns:
            return "left", column
        if column in right_columns:
            return "right", column
        raise Exception("Column does not exist in either table")

    def comparison_type(self, comparison):
        """
        Return the comparison expression
        :param comparison:
        :return:
        """
        return comparison

    def join_expression(self, *args):
        """
        Evaluate a join into one dataframe using a merge method
        :return:
        """
        # There will only ever be four args if a join is specified and three if a
        # join isn't specified
        if len(args) == 3:
            join_type = "inner"
            table1 = args[0]
            table2 = args[1]
            join_condition = args[2]
        else:
            table1 = args[0]
            join_type = args[1]
            table2 = args[2]
            join_condition = args[3]
            if "outer" in join_type:
                match = re.match(r"(?P<type>.*)\souter", join_type)
                if match:
                    join_type = match.group("type")
            if join_type in ("full", "cross"):
                join_type = "outer"

        # Check that there is a column from both sides
        column_comparison = join_condition.children[0].children[0].children
        column1 = str(column_comparison[0].children)
        column2 = str(column_comparison[1].children)

        column1_side, column1 = self.determine_column_side(column1, table1, table2)
        column2_side, column2 = self.determine_column_side(column2, table1, table2)
        if column1_side == column2_side:
            raise Exception("Join columns must be one column from each join table!")
        column1 = self.column_name_map[table1][column1]
        column2 = self.column_name_map[table2][column2]
        if column1_side == "left":
            left_on = column1
            right_on = column2
        else:
            left_on = column2
            right_on = column1

        return Join(
            left_table=table1,
            right_table=table2,
            join_type=join_type,
            left_on=left_on,
            right_on=right_on,
        )

    @staticmethod
    def has_star(column_list: List[str]):
        """
        Returns true if any columns have a star
        :param column_list:
        :return:
        """
        for column_name in column_list:
            if re.match(r"\*", column_name):
                return True
        return False

    @staticmethod
    def handle_non_token_non_tree(query_info: QueryInfo, token, token_pos):
        """
        Handles non token_or_tree non tree items and extracts necessary query
        information from it

        :param query_info: Dictionary of all info about the query
        :param token: Item being handled
        :param token_pos: Ordinal position of the item
        :return:
        """
        query_info.all_names.append(token.final_name)
        query_info.name_order[token.final_name] = token_pos

        if token.typename:
            query_info.conversions[token.final_name] = token.typename

        if isinstance(token, Column):
            query_info.columns.append(token)
            query_info.column_selected[token.name] = True
            # TODO Get rid of collecting this alias information since its part of the
            #  column object
            if token.alias:
                query_info.aliases[token.name] = token.alias

        if isinstance(token, Expression):
            query_info.expressions.append(token)

        if isinstance(token, Aggregate):
            query_info.aggregates[token.final_name] = (
                token.value.final_name,
                token.function,
            )
            if isinstance(token.value, Column) and not query_info.column_selected.get(
                token.value.name
            ):
                query_info.columns.append(token.get_value())
                query_info.column_selected[token.value.get_name()] = True

        if isinstance(token, Literal):
            query_info.literals.append(token)

    def handle_token_or_tree(self, query_info: QueryInfo, token_or_tree, item_pos):
        """
        Handles token and extracts necessary query information from it
        :param query_info: Dictionary of all info about the query
        :param token_or_tree: Item being handled
        :param item_pos: Ordinal position of the token
        :return:
        """
        if isinstance(token_or_tree, Token):
            if token_or_tree.type == "from_expression":
                query_info.frame_names.append(token_or_tree.value)
            elif token_or_tree.type == "group":
                query_info.group_columns.append(token_or_tree.value)
            elif token_or_tree.type == "where_expr":
                query_info.where_expr = token_or_tree.value
        elif isinstance(token_or_tree, Tree):
            if token_or_tree.data == "having_expr":
                query_info.having_expr = token_or_tree
        else:
            self.handle_non_token_non_tree(query_info, token_or_tree, item_pos)

    def select(self, *select_expressions: Tuple[Tree]) -> QueryInfo:
        """
        Forms the final sequence of methods that will be executed
        :param select_expressions:
        :return:
        """

        tables = []
        query_info = QueryInfo()

        for select_expression in select_expressions:
            if isinstance(select_expression, Tree):
                if select_expression.data == "from_expression":
                    tables.append(select_expression.children[0])
                elif select_expression.data == "having_expr":
                    query_info.having_expr = select_expression
                elif select_expression.data == "where_expr":
                    query_info.where_expr = select_expression

        select_expressions_no_boolean_clauses = tuple(
            select_expression
            for select_expression in select_expressions
            if isinstance(select_expression, Tree)
            and select_expression.data not in ("having_expr", "where_expr")
            or not isinstance(select_expression, Tree)
        )

        internal_transformer = InternalTransformer(
            tables,
            self.dataframe_map,
            self.column_name_map,
            self.column_to_dataframe_name,
        )

        select_expressions = internal_transformer.transform(
            Tree("select", select_expressions_no_boolean_clauses)
        ).children

        query_info.internal_transformer = internal_transformer

        if isinstance(select_expressions[0], Token):
            if str(select_expressions[0]) == "distinct":
                query_info.distinct = True
            select_expressions = select_expressions[1:]

        for token_pos, token in enumerate(select_expressions):
            self.handle_token_or_tree(query_info, token, token_pos)

        if query_info.having_expr is not None:
            query_info.having_transformer = HavingTransformer(
                tables,
                query_info.group_columns,
                self.dataframe_map,
                self.column_name_map,
                self.column_to_dataframe_name,
            )

        return query_info

    def cross_join(self, df1: TableExpr, df2: TableExpr):
        """
        Returns the crossjoin between two dataframes
        :param df1: TableExpr1
        :param df2: TableExpr2
        :return: Crossjoined dataframe
        """
        return df1.assign(__=1).merge(df2.assign(__=1), on="__").drop(columns=["__"])

    @staticmethod
    def apply_ibis_aggregation(ibis_column: TableColumn, aggregation: str) -> TableColumn:
        if aggregation in NUMERIC_AGGREGATIONS:
            assert isinstance(ibis_column, NumericColumn)
            if aggregation in AVG_AGGREGATIONS:
                return ibis_column.mean()
            if aggregation in SUM_AGGREGATIONS:
                return ibis_column.sum()
        if aggregation in MAX_AGGREGATIONS:
            return ibis_column.max()
        if aggregation in MIN_AGGREGATIONS:
            return ibis_column.min()
        raise Exception(f"Aggregation {aggregation} not implemented for column of "
                        f"type {ibis_column.type()}")

    def handle_aggregation(self, aggregates, group_columns, table: TableExpr):
        """
        Handles all aggregation operations when translating from dictionary info
        to dataframe
        :param aggregates:
        :param group_columns:
        :param table:
        :return:
        """
        aggregate_ibis_columns = []
        for aggregate_column in aggregates:
            column_to_aggregate, aggregation = aggregates[aggregate_column]
            column = self.apply_ibis_aggregation(
                table.get_column(column_to_aggregate), aggregation=aggregation
            )
            column._name = aggregate_column  # TODO There needs to be a way to do this in ibis without using a protected name
            # TODO Also ibis shouldn't be naming columns with aggreations eg
            #  naming a column "mean"
            aggregate_ibis_columns.append(column)

        if group_columns and not aggregates:
            for column in table.columns:
                if column not in group_columns:
                    raise Exception(
                        f"For column {column} you must either group or "
                        f"provide and aggregation"
                    )
            table = table.distinct()
        elif aggregates and not group_columns:
            table = (
                table.aggregate(aggregate_ibis_columns)
            )
        elif aggregates and group_columns:
            table = table.group_by(group_columns).aggregate(aggregate_ibis_columns)
        return table

    def handle_columns(
        self,
        columns: list,
        aliases: dict,
        first_frame: TableExpr,
        where_expr: Tree,
        internal_transformer: InternalTransformer,
    ):
        """
        Returns frame with appropriately selected and named columns
        :param columns:
        :param aliases:
        :param first_frame:
        :param where_expr: Syntax tree containing where clause
        :param internal_transformer: Transformer to transform the where clauses
        :return:
        """
        where_value = None
        if where_expr is not None:
            where_value_token = internal_transformer.transform(
                where_expr
            )
            where_value = where_value_token.value

        column_names = [column.name for column in columns]
        if self.has_star(column_names):
            new_table = first_frame
            if where_value is not None:
                new_table: TableExpr = first_frame.filter(where_value)
        else:
            column_names = []
            final_names = []
            for column in columns:
                true_column_name = self.column_name_map[column.table][
                    column.name.lower()
                ]
                column_names.append(true_column_name)
                if (
                    aliases.get(true_column_name) is None
                    and true_column_name != column.name
                ):
                    aliases[true_column_name] = column.name

                if column.alias:
                    final_names.append(column.alias)
                else:
                    final_names.append(column.name)

            if where_value is not None:
                new_table = first_frame.loc[where_value, column_names]
            else:
                new_table = first_frame.drop(
                    set(first_frame.columns) - set(column_names)
                )
            if aliases:
                new_table = new_table.relabel(aliases)
        return new_table

    def handle_join(self, join: Join) -> TableExpr:
        """
        Return the dataframe and execution plan resulting from a join
        :param join:
        :return:
        """
        left_table = self.get_table(join.left_table)
        right_table = self.get_table(join.right_table)
        return left_table.join(
            right_table, predicates=(join.left_on, join.right_on), how=join.join_type,
        )

    def to_ibis_table(self, query_info: QueryInfo):
        """
        Returns the dataframe resulting from the SQL query
        :return:
        """
        frame_names = query_info.frame_names
        if not query_info.frame_names:
            raise Exception("No table specified")
        first_frame = self.get_table(frame_names[0])

        if isinstance(first_frame, TableExpr) and not isinstance(
            frame_names[0], Subquery
        ):
            execution_plan = f"{frame_names[0]}"
        elif isinstance(first_frame, Join):
            first_frame, join_plan = self.handle_join(join=first_frame)
        for frame_name in frame_names[1:]:
            next_frame = self.get_table(frame_name)
            first_frame = self.cross_join(first_frame, next_frame)

        new_table: TableExpr = self.handle_columns(
            query_info.columns,
            query_info.aliases,
            first_frame,
            query_info.where_expr,
            query_info.internal_transformer,
        )

        expressions = query_info.expressions
        if expressions:
            for expression in expressions:
                if expression.alias in new_table.columns:
                    new_table = new_table.set_column(expression.alias, expression.value)
                else:
                    new_table = new_table.mutate(
                        expression.value.name(expression.alias)
                    )

        literals = query_info.literals
        if literals:
            for literal in literals:
                if literal.alias in new_table.columns:
                    new_table = new_table.set_column(literal.alias, literal.value)
                else:
                    new_table = new_table.mutate(
                        literal.value.name(literal.alias)
                    )

        conversions = query_info.conversions
        for conversion in conversions:
            new_table = new_table.set_column(
                conversion,
                new_table.get_column(conversion).cast(conversions[conversion]),
            )

        new_table = self.handle_aggregation(
            query_info.aggregates, query_info.group_columns, new_table
        )

        if (
            query_info.having_expr is not None
            and query_info.having_transformer is not None
        ):
            having_eval, having_plan = query_info.having_transformer.transform(
                query_info.having_expr
            )
            new_table = new_table.loc[having_eval.children[0], :]

        if query_info.distinct:
            new_table = new_table.distinct()

        order_by = query_info.order_by
        if order_by:
            new_table = new_table.sort_by(order_by)

        if query_info.limit is not None:
            new_table = new_table.head(query_info.limit)

        return new_table

    def set_expr(self, query_info):
        """
        Return different sql_object with set relational operations performed
        :param query_info:
        :return:
        """
        frame = self.to_ibis_table(query_info)
        return frame

    def union_all(
        self,
        frame1_and_plan: Tuple[TableExpr, str],
        frame2_and_plan: Tuple[TableExpr, str],
    ):
        """
        Return union all of two dataframes
        :param frame1_and_plan: Left dataframe and execution plan
        :param frame2_and_plan: Right dataframe and execution plan
        :return:
        """
        plan = (
            f"concat({frame1_and_plan[1]}, {frame2_and_plan[1]}, "
            f"ignore_index=True).reset_index(drop=True)"
        )

        return (
            concat(
                [frame1_and_plan[0], frame2_and_plan[0]], ignore_index=True
            ).reset_index(drop=True),
            plan,
        )

    def union_distinct(
        self,
        frame1_and_plan: Tuple[TableExpr, str],
        frame2_and_plan: Tuple[TableExpr, str],
    ):
        """
        Return union distinct of two dataframes
        :param frame1_and_plan: Left dataframe and execution plan
        :param frame2_and_plan: Right dataframe and execution plan
        :return:
        """
        plan = (
            f"concat({frame1_and_plan[1]}, {frame2_and_plan[1]}, "
            f"ignore_index=True).drop_duplicates().reset_index(drop=True)"
        )

        return (
            concat([frame1_and_plan[0], frame2_and_plan[0]], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True),
            plan,
        )

    def intersect_distinct(
        self,
        frame1_and_plan: Tuple[TableExpr, str],
        frame2_and_plan: Tuple[TableExpr, str],
    ):
        """
        Return intersection of two dataframes
        :param frame1_and_plan: Left dataframe and execution plan
        :param frame2_and_plan: Right dataframe and execution plan
        :return:
        """
        frame1 = frame1_and_plan[0]
        frame2 = frame2_and_plan[0]

        plan = (
            f"merge(left={frame1_and_plan[1]}, right={frame2_and_plan[1]}, "
            f"on={frame1_and_plan[1]}.columns.to_list()).reset_index(drop=True)"
        )

        return (
            merge(
                left=frame1, right=frame2, how="inner", on=frame1.columns.to_list()
            ).reset_index(drop=True),
            plan,
        )

    def except_distinct(
        self,
        frame1_and_plan: Tuple[TableExpr, str],
        frame2_and_plan: Tuple[TableExpr, str],
    ):
        """
        Return first dataframe excluding everything that's also in the second dataframe,
        no duplicates
        :param frame1_and_plan: Left dataframe and execution plan
        :param frame2_and_plan: Right dataframe and execution plan
        :return:
        """
        frame1 = frame1_and_plan[0]
        frame2 = frame2_and_plan[0]
        plan1 = frame1_and_plan[1]
        plan2 = frame2_and_plan[1]

        plan = (
            f"{plan1}[~{plan1}.isin({plan2}).all(axis=1).drop_duplicates("
            f").reset_index(drop=True)"
        )

        return (
            frame1[~frame1.isin(frame2).all(axis=1)]
            .drop_duplicates()
            .reset_index(drop=True),
            plan,
        )

    def except_all(
        self,
        frame1_and_plan: Tuple[TableExpr, str],
        frame2_and_plan: Tuple[TableExpr, str],
    ):
        """
        Return first dataframe excluding everything that's also in the second dataframe,
        with duplicates
        :param frame1_and_plan: Left dataframe and execution plan
        :param frame2_and_plan: Right dataframe and execution plan
        :return:
        """
        frame1 = frame1_and_plan[0]
        frame2 = frame2_and_plan[0]
        plan1 = frame1_and_plan[1]
        plan2 = frame2_and_plan[1]

        plan = f"{plan1}[~{plan1}.isin({plan2}).all(axis=1)].reset_index(drop=True)"

        return frame1[~frame1.isin(frame2).all(axis=1)].reset_index(drop=True), plan

    def final(self, table):
        """
        Returns the final dataframe
        :param table:
        :return:
        """
        DerivedColumn.reset_expression_count()
        return table
