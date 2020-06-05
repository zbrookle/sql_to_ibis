"""
Module containing all lark internal_transformer classes
"""
import re
from typing import List, Tuple, Set

import ibis
from ibis.expr.types import TableExpr
from lark import Token, Tree, v_args

from sql_to_ibis.exceptions.sql_exception import (
    TableExprDoesNotExist,
    InvalidQueryException,
    AmbiguousColumnException,
)
from sql_to_ibis.parsing.transformers import InternalTransformer, TransformerBaseClass
from sql_to_ibis.query_info import QueryInfo
from sql_to_ibis.sql_objects import (
    Aggregate,
    AmbiguousColumn,
    Value,
    CrossJoin,
    Column,
    DerivedColumn,
    Expression,
    Join,
    JoinBase,
    Literal,
    Subquery,
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
        if self._column_to_dataframe_name.get(column) is None:
            self._column_to_dataframe_name[column] = table
        elif isinstance(self._column_to_dataframe_name[column], AmbiguousColumn):
            self._column_to_dataframe_name[column].tables.add(table)
        else:
            original_table = self._column_to_dataframe_name[column]
            self._column_to_dataframe_name[column] = AmbiguousColumn(
                {original_table, table}
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

    def subquery(self, query_info: QueryInfo, alias: Tree):
        """
        Handle subqueries amd return a subquery object
        :param query_info:
        :param alias:
        :return:
        """
        assert alias.data == "alias_string"
        alias_name = alias.children[0].value
        subquery_value = self.to_ibis_table(query_info)
        self.dataframe_map[alias_name] = subquery_value
        subquery = Subquery(
            name=alias_name, query_info=query_info, value=subquery_value
        )
        self._column_name_map[alias_name] = {}
        for column in subquery.value.columns:
            self.add_column_to_column_to_dataframe_name_map(column.lower(), alias_name)
            self._column_name_map[alias_name][column.lower()] = column
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

        left_columns = self._column_name_map[left_table]
        right_columns = self._column_name_map[right_table]
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
        column1 = self._column_name_map[table1][column1]
        column2 = self._column_name_map[table2][column2]
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

        if isinstance(token, (Column, Literal, Expression)):
            query_info.columns.append(token)

        if isinstance(token, Aggregate):
            query_info.aggregates[token.final_name] = token

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
                query_info.table_names.append(token_or_tree.value)
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
            self._column_name_map,
            self._column_to_dataframe_name,
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

        return query_info

    def cross_join(self, table1: str, table2: str):
        """
        Returns the crossjoin between two dataframes
        :param table1: TableExpr1
        :param table2: TableExpr2
        :return: Crossjoined dataframe
        """
        return CrossJoin(left_table=table1, right_table=table2,)

    def from_item(self, item):
        return item

    @staticmethod
    def format_column_needs_agg_or_group_msg(column):
        return f"For column '{column}' you must either group or provide an aggregation"

    def _get_column_table(self, column: str, available_tables: Set[str]):
        table = self._column_to_dataframe_name[column.lower()]
        available_tables = set(available_tables)
        if isinstance(table, AmbiguousColumn):
            global_table_possiblities = table.tables
            contextual_table_possibilities = [
                table_name
                for table_name in global_table_possiblities
                if table_name in available_tables
            ]
            if len(contextual_table_possibilities) > 1:
                raise AmbiguousColumnException(column, list(available_tables))
            return contextual_table_possibilities[0]
        return table

    def _get_true_column_name(self, column: str, available_tables: Set[str]):
        return self._column_name_map[self._get_column_table(column, available_tables)][
            column
        ]

    def _get_true_column_names(self, columns: List[str], available_tables: List[str]):
        available_tables_set = set(available_tables)
        return [
            self._get_true_column_name(column, available_tables_set)
            for column in columns
        ]

    def handle_aggregation(
        self,
        aggregates,
        group_columns: List[str],
        table: TableExpr,
        having_expr: Tree,
        internal_transformer: InternalTransformer,
        selected_columns: List[Value],
        table_names: List[str],
    ):
        """
        Handles all aggregation operations when translating from dictionary info
        to dataframe
        """
        having = None
        if having_expr:
            having = internal_transformer.transform(having_expr.children[0]).value
        aggregate_ibis_columns = []
        for aggregate_column in aggregates:
            column = aggregates[aggregate_column].value.name(aggregate_column)
            aggregate_ibis_columns.append(column)
        if having is not None and not aggregates:
            for column in table.columns:
                if column not in group_columns:
                    raise InvalidQueryException(
                        self.format_column_needs_agg_or_group_msg(column)
                    )
        if group_columns:
            group_columns = self._get_true_column_names(group_columns, table_names)

        if group_columns and not aggregates:
            for column in table.columns:
                if column not in group_columns:
                    raise InvalidQueryException(
                        self.format_column_needs_agg_or_group_msg(column)
                    )
            table = table.distinct()
        elif aggregates and not group_columns:
            table = table.aggregate(aggregate_ibis_columns, having=having)
        elif aggregates and group_columns:
            table = table.group_by(group_columns)
            if having is not None:
                table = table.having(having)
            table = table.aggregate(aggregate_ibis_columns)

        selected_column_names = {column.get_name() for column in selected_columns}
        non_selected_columns = []
        if group_columns:
            for group_column in group_columns:
                if group_column not in selected_column_names:
                    non_selected_columns.append(group_column)
            table = table.drop(non_selected_columns)

        return table

    def _get_unique_list_maintain_order(self, item_list: list):
        item_set = set()
        new_list = []
        for item in item_list:
            if item not in item_set:
                item_set.add(item)
                new_list.append(item)
        return new_list

    def handle_filtering(
        self,
        ibis_table: TableExpr,
        where_expr: Tree,
        internal_transformer: InternalTransformer,
    ):
        """
        Returns frame with appropriately selected and named columns
        :param ibis_table: Ibis expression table to manipulate
        :param where_expr: Syntax tree containing where clause
        :param internal_transformer: Transformer to transform the where clauses
        :return: Filtered TableExpr
        """
        where_value = None
        if where_expr is not None:
            where_value_token = internal_transformer.transform(where_expr)
            where_value = where_value_token.value
        if where_value is not None:
            return ibis_table.filter(where_value)
        return ibis_table

    def subquery_in(self, column: Tree, subquery: Subquery):
        return Tree("subquery_in", (column, subquery))

    def handle_selection(
        self, ibis_table: TableExpr, columns: List[Value]
    ) -> TableExpr:
        column_mutation = []
        for column in columns:
            if column.get_name() == "*":
                return ibis_table
            column_value = column.get_value().name(column.get_name())
            column_mutation.append(column_value)
        if column_mutation:
            return ibis_table.projection(column_mutation)
        return ibis_table

    def handle_join(self, join: JoinBase) -> TableExpr:
        """
        Return the dataframe and execution plan resulting from a join
        :param join:
        :return:
        """
        left_table = self.get_table(join.left_table)
        right_table = self.get_table(join.right_table)
        if isinstance(join, Join):
            return left_table.join(
                right_table,
                predicates=left_table.get_column(join.left_on)
                == right_table.get_column(join.right_on),
                how=join.join_type,
            )
        if isinstance(join, CrossJoin):
            return ibis.cross_join(left_table, right_table)

    def to_ibis_table(self, query_info: QueryInfo):
        """
        Returns the dataframe resulting from the SQL query
        :return:
        """
        table_names = query_info.table_names
        if not query_info.table_names:
            raise Exception("No table specified")
        first_table = self.get_table(table_names[0])

        if isinstance(first_table, JoinBase):
            first_table = self.handle_join(join=first_table)
        for table_name in table_names[1:]:
            next_frame = self.get_table(table_name)
            first_table = first_table.cross_join(next_frame)

        new_table = self.handle_selection(first_table, query_info.columns)
        new_table = self.handle_filtering(
            new_table, query_info.where_expr, query_info.internal_transformer
        )
        new_table = self.handle_aggregation(
            query_info.aggregates,
            query_info.group_columns,
            new_table,
            query_info.having_expr,
            query_info.internal_transformer,
            query_info.columns,
            query_info.table_names,
        )

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
        self, expr1: TableExpr, expr2: TableExpr,
    ):
        """
        Return union distinct of two TableExpr
        :param expr1: Left TableExpr and execution plan
        :param expr2: Right TableExpr and execution plan
        :return:
        """
        return expr1.union(expr2)

    def union_distinct(
        self, expr1: TableExpr, expr2: TableExpr,
    ):
        """
        Return union distinct of two TableExpr
        :param expr1: Left TableExpr
        :param expr2: Right TableExpr
        :return:
        """
        return expr1.union(expr2, distinct=True)

    def intersect_distinct(self, expr1: TableExpr, expr2: TableExpr):
        """
        Return distinct intersection of two TableExpr
        :param expr1: Left TableExpr
        :param expr2: Right TableExpr
        :return:
        """
        raise NotImplementedError("Waiting on ibis for intersect implementation")

    def except_distinct(self, expr1: TableExpr, expr2: TableExpr):
        """
        Return distinct set difference of two TableExpr
        :param expr1: Left TableExpr
        :param expr2: Right TableExpr
        :return:
        """
        raise NotImplementedError("Waiting on ibis for except implementation")

    def except_all(self, expr1: TableExpr, expr2: TableExpr):
        """
        Return set difference of two TableExpr
        :param expr1: Left TableExpr
        :param expr2: Right TableExpr
        :return:
        """
        raise NotImplementedError("Waiting on ibis for except implementation")

    def final(self, table):
        """
        Returns the final dataframe
        :param table:
        :return:
        """
        DerivedColumn.reset_expression_count()
        return table
