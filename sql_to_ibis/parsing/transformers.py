from __future__ import annotations

from datetime import date, datetime
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import ibis
from ibis.expr.api import NumericColumn
from ibis.expr.types import (
    AnyColumn,
    AnyScalar,
    BooleanValue,
    IntegerColumn,
    NumericScalar,
    TableExpr,
)
from ibis.expr.window import Window as IbisWindow
from lark import Token, Transformer, Tree

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
    COUNT_DISTINCT_AGGREGATIONS,
    MAX_AGGREGATIONS,
    MIN_AGGREGATIONS,
    NUMERIC_AGGREGATIONS,
    SUM_AGGREGATIONS,
)
from sql_to_ibis.sql.enums.rank_functions import RankFunction
from sql_to_ibis.sql.sql_clause_objects import (
    AliasExpression,
    ColumnExpression,
    ExtentExpression,
    Following,
    FrameExpression,
    FromExpression,
    OrderByExpression,
    PartitionByExpression,
    Preceding,
    ValueExpression,
    WhereExpression,
)
from sql_to_ibis.sql.sql_objects import AliasRegistry, AmbiguousColumn, Window
from sql_to_ibis.sql.sql_value_objects import (
    Aggregate,
    Column,
    CountStar,
    Date,
    Expression,
    GroupByColumn,
    Literal,
    NestedCrossJoin,
    NestedJoinBase,
    Number,
    String,
    Subquery,
    Table,
    TableOrJoinbase,
    Value,
)

TableMap = MutableMapping[Union[str, AmbiguousColumn], TableOrJoinbase]


def num_eval(arg: Union[Token, float, int]) -> Union[int, float]:
    """
    Takes an argument that may be a string or number and outputs a number
    :param arg:
    :return:
    """
    if isinstance(arg, Token):
        return eval(arg)
    return arg


class Extents(TypedDict):
    following: Following
    preceding: Preceding


TransformerBaseLeaf = TypeVar("TransformerBaseLeaf")
TransformerBaseReturn = TypeVar("TransformerBaseReturn")


class TransformerBaseClass(Transformer[TransformerBaseLeaf, TransformerBaseReturn]):
    """
    Base class for transformers
    """

    _CURRENT_ROW = "CURRENT ROW"

    def __init__(
        self,
        table_name_map: Dict[str, str],
        table_map: TableMap,
        column_name_map: Dict[str, Dict[str, str]],
        column_to_table_name: Dict[str, Union[str, AmbiguousColumn]],
        _temp_dataframes_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(visit_tokens=False)
        self._table_name_map = table_name_map
        self._table_map: TableMap = {}
        for name, table in table_map.items():
            if isinstance(table, Table):
                self._table_map[name] = table
        self._column_name_map = column_name_map
        self._column_to_table_name = column_to_table_name
        self._temp_dataframes_dict = _temp_dataframes_dict

    def name(self, name_tokens: List[Token]) -> str:
        """
        Cleans the name depending on whether it is in quotes or not
        """
        token = name_tokens[0]
        if token.type == "CNAME":
            return token.value
        return token.value[1:-1]


class InternalTransformer(TransformerBaseClass):
    """
    Evaluates subtrees with knowledge of provided tables that are in the proper scope
    """

    def __init__(
        self,
        tables: List[TableOrJoinbase],
        table_map: TableMap,
        column_name_map: Dict[str, Dict[str, str]],
        column_to_table_name: Dict[str, Union[str, AmbiguousColumn]],
        table_name_map: Dict[str, str],
        alias_registry: AliasRegistry,
    ) -> None:
        super().__init__(
            table_name_map=table_name_map,
            table_map=table_map,
            column_name_map=column_name_map,
            column_to_table_name=column_to_table_name,
        )
        self._tables = tables
        self._table_names_list = [
            table.name for table in tables if isinstance(table, Table)
        ]
        self._column_to_table_name = column_to_table_name.copy()  # This must be
        # copied because when ambiguity is resolved in the following method,
        # we don't want that resolution to carry over to other subqueries
        self._remove_non_selected_tables_from_transformation()
        self._alias_registry = alias_registry

    def set_column_value(self, column: Column, table_name: str = "") -> None:
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
        derived_table_name: Union[str, AmbiguousColumn] = table_name
        if not derived_table_name:
            if lower_column_name not in self._column_to_table_name:
                raise ColumnNotFoundError(column.name, self._table_names_list)
            derived_table_name = self._column_to_table_name[column.name.lower()]
        if isinstance(derived_table_name, AmbiguousColumn):
            raise AmbiguousColumnException(column.name, list(derived_table_name.tables))
        table = self.get_table(derived_table_name)
        table_column_name_map = self._column_name_map[table.name]
        if lower_column_name not in table_column_name_map:
            raise ColumnNotFoundError(column.name, [derived_table_name])
        column_true_name = table_column_name_map[column.name.lower()]
        column.value = table.get_table_expr()[column_true_name]
        column.set_table(table)

    def _remove_non_selected_tables_from_transformation(self) -> None:
        all_selected_table_names = {
            table.name if isinstance(table, Table) else table
            for table in self._table_names_list
        }
        for column in self._column_to_table_name:
            table = self._column_to_table_name[column]
            if isinstance(table, AmbiguousColumn):
                present_tables = [
                    ambiguous_table
                    for ambiguous_table in table.tables
                    if ambiguous_table in all_selected_table_names
                ]
                if len(present_tables) == 1:
                    self._column_to_table_name[column] = present_tables[0]
                elif len(present_tables) > 1:
                    self._column_to_table_name[column] = AmbiguousColumn(
                        set(present_tables)
                    )

    def transform(self, tree: Tree) -> Union[Value, ValueExpression, Tree]:
        new_tree = TransformerBaseClass.transform(self, tree)
        if isinstance(new_tree, Token) and isinstance(new_tree.value, Value):
            new_tree.value = new_tree.value.value
        return new_tree

    def apply_ibis_aggregation(
        self, column: Union[Column, IbisWindow], aggregation: str
    ) -> Union[CountStar, AnyScalar]:
        aggregation = aggregation.replace("(", "")  # Needed for ensuring ( directly
        # follows all aggregate functions
        ibis_column = column
        if isinstance(ibis_column, Column):
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
        if aggregation in COUNT_DISTINCT_AGGREGATIONS:
            return ibis_column.nunique()
        raise UnsupportedColumnOperation(type(ibis_column), aggregation)

    def sql_aggregation(
        self, agg_parts: Tuple[Token, Column, Optional[List[ColumnExpression]]]
    ) -> Value:
        aggregation, column, window_parts = agg_parts
        ibis_aggregation = self.apply_ibis_aggregation(
            column, aggregation.value.lower()
        )
        if isinstance(ibis_aggregation, NumericScalar) and window_parts is not None:
            return Column(
                name=column.name,
                alias=column.alias,
                typename=column.typename,
                value=Window(
                    window_parts, ibis_aggregation
                ).apply_ibis_window_function(),
            )
        return Aggregate(
            ibis_aggregation,
            alias=column.alias,
            typename=column.typename,
        )

    def expression_mul(self, args: Tuple[Value, Value]) -> Value:
        """
        Returns the product of two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 * arg2

    def expression_add(self, args: Tuple[Value, Value]) -> Value:
        """
        Returns the sum of two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 + arg2

    def expression_sub(self, args: Tuple[Value, Value]) -> Value:
        """
        Returns the difference between two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 - arg2

    def expression_div(self, args: Tuple[Value, Value]) -> Value:
        """
        Returns the difference between two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 / arg2

    def number(self, numerical_value: Tuple[Token]) -> Number:
        """
        Return a number token_or_tree with a numeric value as a child
        :param numerical_value:
        :return:
        """
        return Number(num_eval(numerical_value[0]))

    def string(self, string_token: Tuple[Token]) -> String:
        """
        Return value of the token_or_tree associated with the string
        :param string_token:
        :return:
        """
        val = string_token[0].value
        without_quotes = val[1 : len(val) - 1]
        return String(without_quotes)

    def timestamp_expression(self, date_list: List[Date]) -> Date:
        """
        Return a timestamp object
        :param date_list:
        :return:
        """
        return date_list[0]

    @staticmethod
    def int_token_list(token_list: List[Token]) -> List[int]:
        """
        Returns a list of integer from a list of tokens
        :param token_list:
        :return:
        """
        return [int(token.value) for token in token_list]

    def date(self, date_list: List[Token]) -> List[int]:
        """
        Returns list with correct date integers
        :param date_list:
        :return:
        """
        return self.int_token_list(date_list)

    def time(self, time_list: List[Token]) -> List[int]:
        """
        Returns list with correct time integers
        :param time_list:
        :return:
        """
        return self.int_token_list(time_list)

    def custom_timestamp(
        self, datetime_list: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    ) -> Literal:
        """
        Return a custom time stamp based on user input
        :param datetime_list:
        :return:
        """
        return Literal(datetime(*(datetime_list[0] + datetime_list[1])))

    def datetime_now(self, _: Any) -> Literal:
        """
        Return current date and time
        :return:
        """
        date_value = Literal(datetime.now())
        date_value.set_alias("now()")
        return date_value

    def date_today(self, _: Any) -> Literal:
        """
        Return current date
        :return:
        """
        date_value = Literal(date.today())
        date_value.set_alias("today()")
        return date_value

    def not_equals(self, expressions: Tuple[Value, Value]) -> Value:
        """
        Compares two expressions for inequality
        :param expressions:
        :return:
        """
        return Value(expressions[0] != expressions[1])

    def is_null(self, expressions: Tuple[Value]) -> Value:
        return Value(expressions[0]).is_null()

    def is_not_null(self, expressions: Tuple[Value]) -> Value:
        return Value(expressions[0]).is_not_null()

    def greater_than(self, expressions: Tuple[Value, Value]) -> Value:
        """
        Performs a greater than sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] > expressions[1])

    def greater_than_or_equal(self, expressions: Tuple[Value, Value]) -> Value:
        """
        Performs a greater than or equal sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] >= expressions[1])

    def less_than(self, expressions: Tuple[Value, Value]) -> Value:
        """
        Performs a less than sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] < expressions[1])

    def less_than_or_equal(self, expressions: Tuple[Value, Value]) -> Value:
        """
        Performs a less than or equal sql_object
        :param expressions:
        :return:
        """
        return Value(expressions[0] <= expressions[1])

    def between(self, expressions: List[Value]) -> Value:
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

    def _get_expression_values(self, expressions: List[Value]) -> List[AnyColumn]:
        return [expression.get_value() for expression in expressions]

    def in_expr(self, expressions: List[Value]) -> Value:
        """
        Evaluate in sql_object
        :param expressions:
        :return:
        """
        in_list = self._get_expression_values(expressions[1:])
        return Value(expressions[0].value.isin(in_list))

    def not_in_expr(self, expressions: List[Value]) -> Value:
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

    def equals(self, expressions: Tuple[Value, Value]) -> Value:
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
        truth_series_pair_values: List[BooleanValue] = []
        for i, value in enumerate(truth_series_pair):
            truth_series_pair_values.append(value.get_value())
        return Value(
            truth_series_pair_values[0] & truth_series_pair_values[1],
        )

    def bool_parentheses(self, bool_expression_in_list: List[Value]) -> Value:
        return bool_expression_in_list[0]

    def bool_or(self, truth_series_pair: Tuple[Value, Value]) -> Value:
        """
        Return the truth value of the series pair
        :param truth_series_pair:
        :return:
        """
        return Value(truth_series_pair[0] | truth_series_pair[1])

    def where_expr(self, where_value_list: List[Value]) -> WhereExpression:
        """
        Return a where token_or_tree
        :param where_value_list:
        :return: Token
        """
        return WhereExpression(where_value_list[0])

    def alias_string(self, name: List[str]) -> AliasExpression:
        """
        Returns an alias token_or_tree with the name extracted
        :param name:
        :return:
        """
        return AliasExpression(str(name[0]))

    def cross_join_expression(
        self, cross_join_list: List[NestedCrossJoin]
    ) -> NestedCrossJoin:
        return cross_join_list[0]

    def from_expression(
        self, expression: List[Union[Subquery, NestedJoinBase, Table]]
    ) -> FromExpression:
        """
        Return a from sql_object token_or_tree
        :param expression:
        :return: Token from sql_object
        """
        return FromExpression(expression[0])

    def when_then(self, when_then_values: List[Value]) -> Tuple[Value, Value]:
        """
        When / then sql_object
        :param when_then_values:
        :return:
        """
        return when_then_values[0], when_then_values[1]

    def case_expression(
        self, when_expressions: List[Union[Tuple[Value, Value], Value]]
    ) -> Expression:
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

    def window_form(
        self,
        form: List[
            Optional[Union[PartitionByExpression, OrderByExpression, FrameExpression]]
        ],
    ) -> List[
        Optional[Union[PartitionByExpression, OrderByExpression, FrameExpression]]
    ]:
        """
        Returns the window form
        :param form:
        :return:
        """
        return form

    def order_asc(self, column_list: List[Column]) -> OrderByExpression:
        return OrderByExpression(column_list[0])

    def order_desc(self, column_list: List[Column]) -> OrderByExpression:
        return OrderByExpression(column_list[0], False)

    def partition_by(self, column_list: List[Column]) -> PartitionByExpression:
        """
        Returns a partition token_or_tree containing the corresponding column
        :param column_list: List containing only one column
        :return:
        """
        return PartitionByExpression(column_list[0])

    def apply_rank_function(
        self, first_column: AnyColumn, rank_function: RankFunction
    ) -> IntegerColumn:
        """
        :param first_column:
        :param rank_function:
        :return:
        """
        if rank_function == RankFunction.RANK:
            return first_column.rank()
        return first_column.dense_rank()

    def rank(
        self,
        column_clause_list: List[ColumnExpression],
        rank_function: RankFunction,
    ) -> Expression:
        """
        :param column_clause_list_list:
        :param rank_function:
        :return:
        """
        first_column = column_clause_list[0].column.get_value()
        window = Window(column_clause_list, first_column)
        return Expression(
            self.apply_rank_function(first_column, rank_function).over(
                ibis.window(order_by=window.order_by, group_by=window.partition)
            )
        )

    @staticmethod
    def _extract_column_expressions_for_rank(
        tokens: List[List[Optional[ColumnExpression]]],
    ) -> List[ColumnExpression]:
        expressions = tokens[0]
        return [expression for expression in expressions if expression is not None]

    def rank_expression(
        self, tokens: List[List[Optional[ColumnExpression]]]
    ) -> Expression:
        """
        Handles rank expressions
        :param tokens:
        :return:
        """
        return self.rank(
            self._extract_column_expressions_for_rank(tokens), RankFunction.RANK
        )

    def dense_rank_expression(
        self, tokens: List[List[Optional[ColumnExpression]]]
    ) -> Expression:
        """
        Handles dense_rank_expressions
        :param tokens:
        :return:
        """
        return self.rank(
            self._extract_column_expressions_for_rank(tokens), RankFunction.DENSE_RANK
        )

    def coalesce_expression(self, tokens: List[Value]) -> Column:
        coalesce_args = [token.value for token in tokens]
        return Column(ibis.coalesce(*coalesce_args))

    def select_expression(
        self, expression_and_alias: Tuple[Value, Optional[AliasExpression]]
    ) -> Value:
        """
        Returns the appropriate object for the given sql_object
        :param expression_and_alias: An sql_object token_or_tree and
              A token_or_tree containing the name to be assigned
        :return:
        """
        expression = expression_and_alias[0]
        alias_expression = None
        if len(expression_and_alias) == 2:
            alias_expression = expression_and_alias[1]

        if alias_expression:
            expression.set_alias(alias_expression.alias)
        return expression

    def group_by(self, columns: List[Column]) -> GroupByColumn:
        """
        Returns a group token_or_tree
        :param columns: Column to group by
        :return: group token_or_tree
        """
        assert len(columns) == 1
        column = columns[0]
        group_by = GroupByColumn.from_column_type(column)
        return group_by

    def as_type(self, column_and_type: Tuple[Column, Token]) -> Column:
        """
        Extracts token_or_tree type and returns tree object with sql_object and type
        :param column_and_type: Column object and type to cast as
        :return:
        """
        column, typename = column_and_type
        column.set_type(to_ibis_type(typename.value))
        return column

    def literal_cast(self, value_and_type: Tuple[Value, str]) -> Value:
        """
        Cast variable as the given given_type for a literal
        :param value_and_type: Value and pandas dtype to be cast as
        :return:
        """
        value_wrapper, given_type = value_and_type
        new_type = TYPE_TO_SQL_TYPE[given_type]
        new_value = new_type(value_wrapper.value.cast(to_ibis_type(given_type)))
        return new_value

    def subquery_in(self, column_and_subquery: Tuple[Column, Subquery]) -> Value:
        column, subquery = column_and_subquery
        subquery_table = self.get_table(subquery)
        if len(subquery_table.column_names) != 1:
            raise InvalidQueryException(
                "Can only perform 'in' operation on subquery with one column present"
            )
        return Value(
            column.value.isin(
                subquery_table.get_table_expr().get_column(
                    subquery_table.column_names[0]
                )
            )
        )

    def get_table(self, table_or_alias_name: Union[Table, str]) -> Table:
        if isinstance(table_or_alias_name, Table):
            return table_or_alias_name
        try_get_table = self._table_map.get(table_or_alias_name)
        if try_get_table is not None:
            assert isinstance(try_get_table, Table)
            return try_get_table
        if table_or_alias_name not in self._alias_registry:
            raise Exception(f"Table or alias '{table_or_alias_name}' not found")
        return self._alias_registry.get_registry_entry(table_or_alias_name)

    def column_name(self, name_list: List[str]) -> Column:
        """
        Returns a column token_or_tree with the name extracted
        :param name_list: List formatted name
        :return: Tree with column token_or_tree
        """
        name = name_list[0]
        table_name = ""
        if "." in name:
            table_name, name = name.split(".")
            table_name_lower = table_name.lower()
            if table_name_lower in self._table_name_map:
                table_name = self._table_name_map[table_name_lower]
        column = Column(name="".join(name))
        self.set_column_value(column, table_name)
        return column

    def __frame_extract(self, specs: List[Union[Token, int]]) -> Optional[int]:
        value = specs[0]
        if isinstance(value, Token) and value.value == "UNBOUNDED":
            return None
        return cast(int, value)

    def frame_preceding(self, preceding_specs: List[Union[Token, int]]) -> Preceding:
        return Preceding(self.__frame_extract(preceding_specs))

    def frame_following(self, following_specs: List[Union[Token, int]]) -> Following:
        return Following(self.__frame_extract(following_specs))

    def frame_bound(
        self, extent_expression_list: List[ExtentExpression]
    ) -> Union[str, ExtentExpression]:
        if not extent_expression_list:
            return self._CURRENT_ROW
        return extent_expression_list[0]

    def frame_between(
        self,
        extent_expressions: Tuple[Union[Literal, Preceding], Union[Literal, Following]],
    ) -> List[Union[Literal, Preceding, Following]]:
        preceding = extent_expressions[0]
        following = extent_expressions[1]
        if extent_expressions[0] == self._CURRENT_ROW:
            preceding = Preceding(0)
        if extent_expressions[1] == self._CURRENT_ROW:
            following = Following(0)
        # TODO: Switch this to a tuple (causing errors to do that currently)
        return [preceding, following]

    def frame_extent(self, extent_list: List[ExtentExpression]) -> Extents:
        if isinstance(extent_list[0], list):
            extent_list = extent_list[0]
        extents = Extents(following=Following(), preceding=Preceding())
        for extent in extent_list:
            if isinstance(extent, Following):
                extents["following"] = extent
            if isinstance(extent, Preceding):
                extents["preceding"] = extent
        return extents

    def row_range_clause(self, clause: Tuple[Token, Extents]) -> FrameExpression:
        return FrameExpression(clause[0].value.lower(), **clause[1])


class InternalTransformerWithStarVal(InternalTransformer):
    def __init__(
        self,
        tables: List[TableOrJoinbase],
        table_map: TableMap,
        column_name_map: Dict[str, Dict[str, str]],
        column_to_table_name: Dict[str, Union[str, AmbiguousColumn]],
        table_name_map: Dict[str, str],
        alias_registry: AliasRegistry,
        available_relations: List[TableExpr],
    ) -> None:
        super().__init__(
            tables=tables,
            table_name_map=table_name_map,
            table_map=table_map,
            column_name_map=column_name_map,
            column_to_table_name=column_to_table_name,
            alias_registry=alias_registry,
        )
        self._available_relations = available_relations

    def set_column_value(self, column: Column, table_name: str = "") -> None:
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
    ) -> InternalTransformerWithStarVal:
        return cls(
            internal_transformer._tables,
            internal_transformer._table_map,
            internal_transformer._column_name_map,
            internal_transformer._column_to_table_name,
            internal_transformer._table_name_map,
            internal_transformer._alias_registry,
            available_relations,
        )

    def apply_ibis_aggregation(
        self, column: Column, aggregation: str
    ) -> Union[CountStar, AnyScalar]:
        if aggregation in COUNT_AGGREGATIONS and column.name == "*":
            table = column.get_table()
            assert table is not None
            return table.get_table_expr().count()
        return super().apply_ibis_aggregation(column, aggregation)
