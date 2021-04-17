from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from lark import Tree

from sql_to_ibis.parsing.transformers import InternalTransformer
from sql_to_ibis.sql.sql_clause_objects import FromExpression
from sql_to_ibis.sql.sql_value_objects import (
    Aggregate,
    Column,
    Expression,
    GroupByColumn,
    Literal,
    NestedJoinBase,
    Subquery,
    Table,
    Value,
)


@dataclass
class InSubqueryInfo:
    column_tree: Tree
    subquery: Subquery


@dataclass
class OrderByInfo:
    column_name: str
    ascending: bool

    def get_tuple(self) -> Tuple[str, bool]:
        return self.column_name, self.ascending


class QueryInfo:
    """
    Class that holds metadata extracted / derived from a sql query
    """

    def __init__(
        self,
        internal_transformer: InternalTransformer,
        select_expressions_no_boolean_clauses: Optional[List[Union[str, Tree]]] = None,
        having_expr=None,
        where_expr=None,
        distinct: bool = False,
    ):
        self.columns: List[Value] = []
        self.tables: List[Union[Table, NestedJoinBase]] = []
        self.all_names: List[str] = []
        self.name_order: Dict[str, int] = {}
        self.aggregates: Dict[str, Aggregate] = {}
        self.group_columns: List[GroupByColumn] = []
        self.where_expr = where_expr
        self.having_expr = having_expr
        self.internal_transformer: InternalTransformer = internal_transformer
        self.order_by: List[Tuple[str, bool]] = []
        self.limit: Optional[int] = None
        self.distinct = distinct
        self.select_expressions_no_boolean_clauses = (
            select_expressions_no_boolean_clauses
            if select_expressions_no_boolean_clauses is not None
            else []
        )

    def add_table(self, table: Union[Table, NestedJoinBase]):
        self.tables.append(table)

    def add_column(self, column: Value):
        self.columns.append(column)

    def add_order_by_info(self, order_by_info: OrderByInfo):
        self.order_by.append(order_by_info.get_tuple())

    def __handle_token_or_tree(self, token_or_tree, item_pos: int):
        """
        Handles token and extracts necessary query information from it
        :param token_or_tree: Item being handled
        :param item_pos: Ordinal position of the token
        :return:
        """
        if isinstance(token_or_tree, FromExpression):
            self.add_table(token_or_tree.value)
        else:
            self.__handle_non_token_non_tree(token_or_tree, item_pos)

    def __handle_non_token_non_tree(self, token, token_pos: int):
        """
        Handles non token_or_tree non tree items and extracts necessary query
        information from it

        :param token: Item being handled
        :param token_pos: Ordinal position of the item
        :return:
        """
        self.all_names.append(token.final_name)
        self.name_order[token.final_name] = token_pos

        if isinstance(token, GroupByColumn):
            self.group_columns.append(token)
        elif isinstance(token, (Column, Literal, Expression)):
            self.add_column(token)
        elif isinstance(token, Aggregate):
            self.aggregates[token.final_name] = token

    def __get_internal_transformer_select_expression(self):
        return self.internal_transformer.transform(
            Tree("select", self.select_expressions_no_boolean_clauses)
        ).children

    def __handle_tokens_and_trees_in_select_expressions(self, select_expressions):
        for token_pos, token in enumerate(select_expressions):
            self.__handle_token_or_tree(token, token_pos)

    def perform_transformation(self):
        select_expressions = self.__get_internal_transformer_select_expression()
        self.__handle_tokens_and_trees_in_select_expressions(select_expressions)
        self.sanitize_order_by_clause()

    def sanitize_order_by_clause(self):
        """
        Names in the order by clause must be cleansed based on info provided by the
        internal transformer
        """
        for i, order_by_column_tuple in enumerate(self.order_by):
            column_list = [order_by_column_tuple[0]]
            column = self.internal_transformer.column_name(column_list)
            ascending = order_by_column_tuple[1]
            self.order_by[i] = (column.get_value(), ascending)

    def __repr__(self):
        return (
            f"Query Information\n"
            f"-----------------\n"
            f"Columns: {self.columns}\n"
            f"Tables: {self.tables}\n"
            f"All names: {self.all_names}\n"
            f"Name order: {self.name_order}\n"
            f"Aggregates: {self.aggregates}\n"
            f"Group columns: {self.group_columns}\n"
            f"Order by: {self.order_by}\n"
            f"Limit: {self.limit}"
        )
