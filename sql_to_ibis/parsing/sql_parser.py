"""
Module containing all lark internal_transformer classes
"""
import re
from typing import List, Tuple

import ibis
from ibis.expr.types import TableExpr
from lark import Token, Tree, v_args
from pandas import merge

from sql_to_ibis.exceptions.sql_exception import TableExprDoesNotExist
from sql_to_ibis.parsing.transformers import InternalTransformer, TransformerBaseClass
from sql_to_ibis.query_info import QueryInfo
from sql_to_ibis.sql_objects import (
    Aggregate,
    AmbiguousColumn,
    Column,
    CrossJoin,
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
            query_info.aggregates[token.final_name] = token

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

        return query_info

    def cross_join(self, table1: str, table2: str):
        """
        Returns the crossjoin between two dataframes
        :param table1: TableExpr1
        :param table2: TableExpr2
        :return: Crossjoined dataframe
        """
        return CrossJoin(left_table=table1, right_table=table2,)

    def handle_aggregation(
        self,
        aggregates,
        group_columns,
        table: TableExpr,
        having_expr: Tree,
        internal_transformer: InternalTransformer,
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
        if group_columns and not aggregates:
            for column in table.columns:
                if column not in group_columns:
                    raise Exception(
                        f"For column {column} you must either group or "
                        f"provide and aggregation"
                    )
            table = table.distinct()
        elif aggregates and not group_columns:
            # print(aggregate_ibis_columns)
            table = table.aggregate(aggregate_ibis_columns, having=having)
        elif aggregates and group_columns:
            table = table.group_by(group_columns)
            if having is not None:
                table = table.having(having)
            table = table.aggregate(aggregate_ibis_columns)
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
        :return:
        """
        where_value = None
        if where_expr is not None:
            where_value_token = internal_transformer.transform(where_expr)
            print(where_value_token)
            where_value = where_value_token.value

        if where_value is not None:
            print(where_value)
            return ibis_table.filter(where_value)
        return ibis_table

    def handle_selection(
        self, ibis_table: TableExpr, columns: List[Column]
    ) -> TableExpr:
        column_mutation = []
        for column in columns:
            if column.name == "*":
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
        frame_names = query_info.frame_names
        if not query_info.frame_names:
            raise Exception("No table specified")
        first_frame = self.get_table(frame_names[0])

        if isinstance(first_frame, JoinBase):
            first_frame = self.handle_join(join=first_frame)
        for frame_name in frame_names[1:]:
            next_frame = self.get_table(frame_name)
            first_frame = first_frame.cross_join(next_frame)

        new_table = self.handle_selection(first_frame, query_info.columns)
        new_table = self.handle_filtering(
            new_table, query_info.where_expr, query_info.internal_transformer
        )

        ibis_expressions = []
        expressions = query_info.expressions
        for expression in expressions:
            value = expression.value
            if expression.alias in new_table.columns:
                value = value.name(expression.alias)
            else:
                value = value.name(expression.final_name)
            ibis_expressions.append(value)

        literals = query_info.literals
        for literal in literals:
            value = literal.value
            if literal.alias:
                value = value.name(literal.alias)
            else:
                value = value.name(literal.final_name)
            ibis_expressions.append(value)

        conversions = query_info.conversions
        for conversion in conversions:
            ibis_expressions.append(
                new_table.get_column(conversion)
                .cast(conversions[conversion])
                .name(conversion)
            )

        if ibis_expressions:
            new_table = new_table.mutate(ibis_expressions)

        new_table = self.handle_aggregation(
            query_info.aggregates,
            query_info.group_columns,
            new_table,
            query_info.having_expr,
            query_info.internal_transformer,
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
        :param expr1: Left TableExpr and execution plan
        :param expr2: Right TableExpr and execution plan
        :return:
        """
        return expr1.union(expr2, distinct=True)

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
