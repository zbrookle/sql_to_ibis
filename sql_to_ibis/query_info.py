from typing import List, Optional, Tuple

from sql_to_ibis.parsing.transformers import InternalTransformer


class QueryInfo:
    """
    Class that holds metadata extracted / derived from a sql query
    """

    def __init__(self):
        self.column_selected = {}
        self.columns = []
        self.expressions = []
        self.literals = []
        self.frame_names = []
        self.aliases = {}
        self.all_names = []
        self.name_order = {}
        self.conversions = {}
        self.aggregates = {}
        self.group_columns = []
        self.where_expr = None
        self.distinct = False
        self.having_expr = None
        self.internal_transformer: Optional[InternalTransformer] = None
        self.order_by: List[Tuple[str, bool]] = []
        self.limit: Optional[int] = None

    @staticmethod
    def set_none_var(value, default):
        return default if not value else value
