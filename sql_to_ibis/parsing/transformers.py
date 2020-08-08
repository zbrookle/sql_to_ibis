from datetime import date, datetime
from typing import Dict, List, Tuple, Union

import ibis
from ibis.expr.api import NumericColumn
from ibis.expr.operations import TableColumn
from ibis.expr.types import ColumnExpr, TableExpr
from lark import Token, Transformer, Tree
from pandas import Series

from sql_to_ibis.conversions.conversions import TYPE_TO_SQL_TYPE, to_ibis_type
from sql_to_ibis.exceptions.sql_exception import (
    AmbiguousColumnException,
    ColumnNotFoundError,
    InvalidQueryException,
    UnsupportedColumnOperation,
)
from sql_to_ibis.parsing.aggregation_aliases import (
    AVG_AGGREGATIONS,
    COUNT_AGGREGATIONS,
    MAX_AGGREGATIONS,
    MIN_AGGREGATIONS,
    NUMERIC_AGGREGATIONS,
    SUM_AGGREGATIONS,
)
from sql_to_ibis.sql_objects import (
    Aggregate,
    AliasRegistry,
    AmbiguousColumn,
    Column,
    CountStar,
    CrossJoin,
    Date,
    Expression,
    GroupByColumn,
    JoinBase,
    Literal,
    Number,
    String,
    Subquery,
    Table,
    Value,
)


def num_eval(arg):
    """
    Takes an argument that may be a string or number and outputs a number
    :param arg:
    :return:
    """
    assert isinstance(arg, (Token, float, int))
    if isinstance(arg, str):
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


class TransformerBaseClass(Transformer):
    """
    Base class for transformers
    """

    def __init__(
        self,
        table_name_map: Dict[str, str],
        table_map: Dict[str, Table],
        column_name_map: Dict[str, Dict[str, str]],
        column_to_table_name: Dict[str, Union[str, AmbiguousColumn]],
        _temp_dataframes_dict=None,
    ):
        super().__init__(visit_tokens=False)
        self._table_name_map = table_name_map
        self._table_map = table_map
        self._column_name_map = column_name_map
        self._column_to_table_name = column_to_table_name
        self._temp_dataframes_dict = _temp_dataframes_dict


class InternalTransformer(TransformerBaseClass):
    """
    Evaluates subtrees with knowledge of provided tables that are in the proper scope
    """

    def __init__(
        self,
        tables: List[Table],
        table_map: Dict[str, Table],
        column_name_map: Dict[str, Dict[str, str]],
        column_to_table_name: Dict[str, Union[str, AmbiguousColumn]],
        table_name_map: Dict[str, str],
        alias_registry: AliasRegistry,
    ):
        super().__init__(
            table_name_map=table_name_map,
            table_map=table_map,
            column_name_map=column_name_map,
            column_to_table_name=column_to_table_name,
        )
        self._tables = tables
        self._table_names_list = [table.name for table in tables]
        self._column_to_table_name = column_to_table_name.copy()  # This must be
        # copied because when ambiguity is resolved in the following method,
        # we don't want that resolution to carry over to other subqueries
        self._remove_non_selected_tables_from_transformation()
        self._alias_registry = alias_registry

    def set_column_value(
        self, column: Column, table_name: Union[str, AmbiguousColumn] = ""
    ) -> None:
        """
        Sets the column value based on what it is in the table
        :param column:
        :param table_name: Optional, only used if provided
        :return:
        """
        if column.name == "*" and table_name:
            column.set_table(self.get_table(table_name))
            return
        if column.name == "*":
            return
        lower_column_name = column.name.lower()
        if not table_name:
            if lower_column_name not in self._column_to_table_name:
                raise ColumnNotFoundError(column.name, self._table_names_list)
            table_name = self._column_to_table_name[column.name.lower()]
        if isinstance(table_name, AmbiguousColumn):
            raise AmbiguousColumnException(column.name, list(table_name.tables))
        table = self.get_table(table_name)
        table_column_name_map = self._column_name_map[table.name]
        if lower_column_name not in table_column_name_map:
            raise ColumnNotFoundError(column.name, [table_name])
        column_true_name = table_column_name_map[column.name.lower()]
        column.value = table.get_table_expr()[column_true_name]
        column.set_table(table)

    def _remove_non_selected_tables_from_transformation(self):
        all_selected_table_names = {
            table.name if isinstance(table, Table) else table
            for table in self._table_names_list
        }
        for column in self._column_to_table_name:
            table = self._column_to_table_name[column]
            if isinstance(table, AmbiguousColumn):
                present_tables = [
                    ambig_table
                    for ambig_table in table.tables
                    if ambig_table in all_selected_table_names
                ]
                if len(present_tables) == 1:
                    self._column_to_table_name[column] = present_tables[0]
                elif len(present_tables) > 1:
                    self._column_to_table_name[column] = AmbiguousColumn(
                        set(present_tables)
                    )

    def transform(self, tree):
        new_tree = TransformerBaseClass.transform(self, tree)
        if isinstance(new_tree, Token) and isinstance(new_tree.value, Value):
            new_tree.value = new_tree.value.value
        return new_tree

    def apply_ibis_aggregation(
        self, column: Column, aggregation: str
    ) -> Union[CountStar, TableColumn]:
        ibis_column = column.value
        if column.name == "*":
            return CountStar()
        if aggregation in NUMERIC_AGGREGATIONS:
            if not isinstance(ibis_column, NumericColumn):
                raise UnsupportedColumnOperation(type(ibis_column), aggregation)
            if aggregation in AVG_AGGREGATIONS:
                return ibis_column.mean()
            if aggregation in SUM_AGGREGATIONS:
                return ibis_column.sum()
        if aggregation in MAX_AGGREGATIONS:
            return ibis_column.max()
        if aggregation in MIN_AGGREGATIONS:
            return ibis_column.min()
        if aggregation in COUNT_AGGREGATIONS:
            return ibis_column.count()
        raise UnsupportedColumnOperation(type(ibis_column), aggregation)

    def sql_aggregation(self, agg_and_column: list):
        aggregation: Token = agg_and_column[0]
        column: Column = agg_and_column[1]
        return Aggregate(
            self.apply_ibis_aggregation(column, aggregation.value.lower()),
            alias=column.alias,
            typename=column.typename,
        )

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

    def timestamp_expression(self, date_list: List[Date]) -> Date:
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
        return Literal(datetime(*(datetime_list[0] + datetime_list[1])))

    def datetime_now(self, *extra_args):
        """
        Return current date and time
        :param extra_args: Arguments that lark parser must pass in
        :return:
        """
        date_value = Literal(datetime.now())
        date_value.set_alias("now()")
        return date_value

    def date_today(self, *extra_args):
        """
        Return current date
        :param extra_args: Arguments that lark parser must pass in
        :return:
        """
        date_value = Literal(date.today())
        date_value.set_alias("today()")
        return date_value

    def not_equals(self, expressions):
        """
        Compares two expressions for inequality
        :param expressions:
        :return:
        """
        return Value(expressions[0] != expressions[1])

    def greater_than(self, expressions):
        """
        Performs a greater than sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] > expressions[1])

    def greater_than_or_equal(self, expressions):
        """
        Performs a greater than or equal sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] >= expressions[1])

    def less_than(self, expressions):
        """
        Performs a less than sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] < expressions[1])

    def less_than_or_equal(self, expressions):
        """
        Performs a less than or equal sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] <= expressions[1])

    def between(self, expressions: List[Value]):
        """
        Performs a less than or equal and greater than or equal
        :param expressions:
        :return:
        """
        main_expression = expressions[0]
        between_expressions = expressions[1:]
        return Value(
            main_expression.value.between(
                between_expressions[0].value, between_expressions[1].value
            )
        )

    def _get_expression_values(self, expressions: List[Value]):
        return [expression.get_value() for expression in expressions]

    def in_expr(self, expressions: List[Value]):
        """
        Evaluate in sql_object
        :param expressions:
        :return:
        """
        in_list = self._get_expression_values(expressions[1:])
        return Value(expressions[0].value.isin(in_list))

    def not_in_expr(self, expressions: List[Value]):
        """
        Negate in expr
        :param expressions:
        :return:
        """
        not_in_list = self._get_expression_values(expressions[1:])
        return Value(expressions[0].value.notin(not_in_list))

    def bool_expression(self, expression: List[Value]) -> Value:
        """
        Return the bool sql_object
        :param expression:
        :return: boolean sql_object
        """
        return expression[0]

    def equals(self, expressions):
        """
        Compares two expressions for equality
        :param expressions:
        :return:
        """
        return Value(expressions[0] == expressions[1])

    def bool_and(self, truth_series_pair: List[Value]) -> Value:
        """
        Return the truth value of the series pair
        :param truth_series_pair:
        :return:
        """
        truth_series_pair_values: List[Series] = []
        for i, value in enumerate(truth_series_pair):
            truth_series_pair_values.append(value.get_value())

        return Value(truth_series_pair_values[0] & truth_series_pair_values[1],)

    def bool_parentheses(self, bool_expression_in_list: list):
        return bool_expression_in_list[0]

    def bool_or(self, truth_series_pair):
        """
        Return the truth value of the series pair
        :param truth_series_pair:
        :return:
        """
        return Value(truth_series_pair[0] | truth_series_pair[1])

    def comparison_type(self, comparison):
        """
        Return the comparison

        :param comparison:
        :return:
        """
        return comparison[0]

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

    def cross_join_expression(self, cross_join_list: List[CrossJoin]):
        return cross_join_list[0]

    def from_expression(self, expression):
        """
        Return a from sql_object token_or_tree
        :param expression:
        :return: Token from sql_object
        """
        expression = expression[0]
        if isinstance(expression, Tree):
            expression = expression.children[0]
        if isinstance(expression, (Subquery, JoinBase, Table)):
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
        case_expression = ibis.case()
        for i, when_expression in enumerate(when_expressions):
            if isinstance(when_expression, tuple):
                conditional_boolean = when_expression[0].get_value()
                conditional_value = when_expression[1].get_value()
                case_expression = case_expression.when(
                    conditional_boolean, conditional_value
                )
            else:
                case_expression = case_expression.else_(
                    when_expression.get_value()
                ).end()

        return Expression(value=case_expression)

    def rank_form(self, form):
        """
        Returns the rank form
        :param form:
        :return:
        """
        return form

    def order_asc(self, column_list: List[Column]):
        """
        Return sql_object in asc order
        :param column:
        :return:
        """
        return Token("order", (column_list[0], True))

    def order_desc(self, column):
        """
        Return sql_object in asc order
        :param column:
        :return:
        """
        column = column[0]
        return Token("order", (column, False))

    def partition_by(self, column_list):
        """
        Returns a partition token_or_tree containing the corresponding column
        :param column_list: List containing only one column
        :return:
        """
        column = column_list[0]
        return Token("partition", column)

    def _get_rank_orders_and_partitions(self, tokens: List[List[Token]]):
        """
        Returns the evaluated rank expressions
        :param tokens: Tokens making up the rank sql_object
        :return:
        """
        expressions = tokens[0]
        order_list = []
        partition_list = []
        rank_column = None
        for token in expressions:
            if token.type == "order":
                token_tuple: Tuple[Column, bool] = token.value
                ibis_value: ColumnExpr = token_tuple[0].get_value()
                if rank_column is None:
                    rank_column = ibis_value
                if not token_tuple[1]:
                    ibis_value = ibis.desc(ibis_value)
                order_list.append(ibis_value)
            elif token.type == "partition":
                column: Column = token.value
                partition_list.append(column.get_value())
        return order_list, partition_list, rank_column

    def apply_rank_function(self, first_column: ColumnExpr, rank_function: str):
        assert rank_function in {"rank", "dense_rank"}
        if rank_function == "rank":
            return first_column.rank()
        if rank_function == "dense_rank":
            return first_column.dense_rank()

    def rank(self, tokens: List[Token], rank_function: str):
        orders, partitions, first_column = self._get_rank_orders_and_partitions(tokens)
        return Expression(
            self.apply_rank_function(first_column, rank_function).over(
                ibis.window(order_by=orders, group_by=partitions)
            )
        )

    def rank_expression(self, tokens):
        """
        Handles rank expressions
        :param tokens:
        :return:
        """
        return self.rank(tokens, "rank")

    def dense_rank_expression(self, tokens):
        """
        Handles dense_rank_expressions
        :param tokens:
        :return:
        """
        return self.rank(tokens, "dense_rank")

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

    def group_by(self, columns: List[Column]):
        """
        Returns a group token_or_tree
        :param columns: Column to group by
        :return: group token_or_tree
        """
        assert len(columns) == 1
        column = columns[0]
        group_by = GroupByColumn.from_column_type(column)
        return group_by

    def as_type(self, column_and_type):
        """
        Extracts token_or_tree type and returns tree object with sql_object and type
        :param column_and_type: Column object and type to cast as
        :return:
        """
        column: Column = column_and_type[0]
        typename: Token = column_and_type[1]
        column.set_type(to_ibis_type(typename.value))
        return column

    def literal_cast(self, value_and_type: list):
        """
        Cast variable as the given given_type for a literal
        :param value_and_type: Value and pandas dtype to be cast as
        :return:
        """
        value_wrapper, given_type = value_and_type
        new_type = TYPE_TO_SQL_TYPE[given_type]
        new_value = new_type(value_wrapper.value.cast(to_ibis_type(given_type)))
        return new_value

    def subquery_in(self, column_and_subquery):
        column: Column = column_and_subquery[0]
        subquery: Subquery = column_and_subquery[1]
        subquery_table = self.get_table(subquery)
        if len(subquery_table.column_names) != 1:
            raise InvalidQueryException(
                "Can only perform 'in' operation on subquery with one column present"
            )
        return column.value.isin(
            subquery_table.get_table_expr().get_column(subquery_table.column_names[0])
        )

    def get_table(self, table_or_alias_name) -> Table:
        if isinstance(table_or_alias_name, Table):
            return table_or_alias_name
        try_get_table = self._table_map.get(table_or_alias_name)
        if try_get_table is not None:
            return try_get_table
        if try_get_table is None and table_or_alias_name not in self._alias_registry:
            raise Exception(f"Table or alias '{table_or_alias_name}' not found")
        return self._alias_registry.get_registry_entry(table_or_alias_name)

    def column_name(self, name_list_format: List[str]):
        """
        Returns a column token_or_tree with the name extracted
        :param name_list_format: List formatted name
        :return: Tree with column token_or_tree
        """
        name = "".join(name_list_format)
        table_name = ""
        if "." in name:
            table_name, name = name.split(".")
            table_name_lower = table_name.lower()
            if table_name_lower in self._table_name_map:
                table_name = self._table_name_map[table_name_lower]
        column = Column(name="".join(name))
        self.set_column_value(column, table_name)
        return column

    @classmethod
    def empty_transformer(cls):
        return cls([], {}, {}, {}, {}, AliasRegistry())


class InternalTransformerWithStarVal(InternalTransformer):
    def __init__(
        self,
        tables: List[Table],
        table_map: Dict[str, Table],
        column_name_map: Dict[str, Dict[str, str]],
        column_to_table_name: Dict[str, Union[str, AmbiguousColumn]],
        table_name_map: Dict[str, str],
        alias_registry: AliasRegistry,
        available_relations: List[TableExpr],
    ):
        super().__init__(
            tables=tables,
            table_name_map=table_name_map,
            table_map=table_map,
            column_name_map=column_name_map,
            column_to_table_name=column_to_table_name,
            alias_registry=alias_registry,
        )
        self._available_relations = available_relations

    def set_column_value(
        self, column: Column, table_name: Union[str, AmbiguousColumn] = ""
    ) -> None:
        if column.name == "*" and not table_name:
            if len(self._available_relations) > 1:
                raise AmbiguousColumnException(column.name, self._table_names_list)
            column.set_table(self.get_table(self._available_relations[0]))
            return
        super().set_column_value(column, table_name)

    @classmethod
    def from_internal_transformer(
        cls,
        internal_transformer: InternalTransformer,
        available_relations: List[TableExpr],
    ):
        print(internal_transformer._alias_registry)

        return cls(
            internal_transformer._tables,
            internal_transformer._table_map,
            internal_transformer._column_name_map,
            internal_transformer._column_to_table_name,
            internal_transformer._table_name_map,
            internal_transformer._alias_registry,
            available_relations,
        )

    def apply_ibis_aggregation(self, column: Column, aggregation: str) -> TableColumn:
        if aggregation in COUNT_AGGREGATIONS and column.name == "*":
            return column.get_table().get_table_expr().count()
        return super().apply_ibis_aggregation(column, aggregation)
