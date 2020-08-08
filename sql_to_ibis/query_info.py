from typing import Dict, List, Optional, Tuple, Union

from lark import Token, Tree

from sql_to_ibis.parsing.transformers import InternalTransformer
from sql_to_ibis.sql_objects import (
    Aggregate,
    Column,
    Expression,
    GroupByColumn,
    JoinBase,
    Literal,
    Table,
    Value,
)


class QueryInfo:
    """
    Class that holds metadata extracted / derived from a sql query
    """

    def __init__(
        self,
        internal_transformer: InternalTransformer,
        select_expressions_no_boolean_clauses: Optional[tuple] = None,
        having_expr=None,
        where_expr=None,
    ):
        self.columns: List[Value] = []
        self.tables: List[Union[Table, JoinBase]] = []
        self.all_names: List[str] = []
        self.name_order: Dict[str, int] = {}
        self.aggregates: Dict[str, Aggregate] = {}
        self.group_columns: List[GroupByColumn] = []
        self.where_expr = where_expr
        self.having_expr = having_expr
        self.internal_transformer: InternalTransformer = internal_transformer
        self.order_by: List[Tuple[str, bool]] = []
        self.limit: Optional[int] = None
        self.distinct = False
        self.select_expressions_no_boolean_clauses = (
            select_expressions_no_boolean_clauses
            if select_expressions_no_boolean_clauses is not None
            else ()
        )

    def add_table(self, table: Union[Table, JoinBase]):
        self.tables.append(table)

    def add_column(self, column: Value):
        self.columns.append(column)

    def __handle_token_or_tree(self, token_or_tree, item_pos):
        """
        Handles token and extracts necessary query information from it
        :param token_or_tree: Item being handled
        :param item_pos: Ordinal position of the token
        :return:
        """
        if isinstance(token_or_tree, Token):
            if token_or_tree.type == "from_expression":
                self.add_table(token_or_tree.value)
            elif token_or_tree.type == "where_expr":
                self.where_expr = token_or_tree.value
        elif isinstance(token_or_tree, Tree):
            if token_or_tree.data == "having_expr":
                self.having_expr = token_or_tree
        else:
            self.__handle_non_token_non_tree(token_or_tree, item_pos)

    def __handle_non_token_non_tree(self, token, token_pos):
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

    def __extract_distinct_property(self, select_expressions):
        if select_expressions and isinstance(select_expressions[0], Token):
            if str(select_expressions[0]) == "distinct":
                self.distinct = True
            return select_expressions[1:]
        return select_expressions

    def __handle_tokens_and_trees_in_select_expressions(self, select_expressions):
        for token_pos, token in enumerate(select_expressions):
            self.__handle_token_or_tree(token, token_pos)

    def perform_transformation(self):
        select_expressions = self.__get_internal_transformer_select_expression()
        print(select_expressions)
        select_expressions = self.__extract_distinct_property(select_expressions)
        self.__handle_tokens_and_trees_in_select_expressions(select_expressions)

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
