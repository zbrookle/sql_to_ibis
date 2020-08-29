import re
from typing import Optional, Union, ClassVar

import ibis
from ibis.expr.types import AnyColumn, AnyScalar, TableExpr, ValueExpr
from pandas import Series
from dataclasses import dataclass, InitVar


@dataclass(unsafe_hash=True)
class Table:
    value: InitVar[TableExpr]
    name: str
    alias: str = ""

    def __post_init__(self, value: TableExpr):
        self._value = value

    def get_table_expr(self):
        return self._value

    def get_ibis_columns(self):
        return self._value.get_columns(self.column_names)

    def get_alias_else_name(self):
        if self.alias:
            return self.alias
        return self.name

    @property
    def column_names(self):
        return self._value.columns


@dataclass
class Value:
    """
    Parent class for expression_count and columns
    """
    value: AnyColumn
    alias: str = ""
    typename: str = ""

    def __post_init__(self):
        self.final_name = self.alias

    def get_value_repr(self):
        string_type = str(type(self.value))
        ibis_repr = ""
        if isinstance(self.value, ValueExpr):
            match = re.match(
                r"<class 'ibis\.expr\.types\.(?P<ibis_type>\w+)'>", string_type
            )
            if match:
                ibis_repr = match.group("ibis_type")
        if ibis_repr:
            return "Ibis" + ibis_repr + "()"
        return self.value

    def __repr__(self):
        print_value = self.get_value_repr()

        display = (
            f"{type(self).__name__}(final_name={self.final_name}, value={print_value}"
        )
        if self.alias:
            display += f", alias={self.alias}"
        if self.typename:
            display += f", type={self.typename}"
        return display

    def __add__(self, other):
        return Expression(
            value=self.value + self.get_other_value(other), alias=self.alias,
        )

    def __sub__(self, other):
        return Expression(
            value=self.value - self.get_other_value(other), alias=self.alias,
        )

    def __mul__(self, other):
        return Expression(
            value=self.value * self.get_other_value(other), alias=self.alias,
        )

    def __truediv__(self, other):
        return Expression(
            value=self.value / self.get_other_value(other), alias=self.alias,
        )

    def get_table(self):
        """
        Returns the table of the current value
        :return:
        """
        return None

    def get_name(self) -> str:
        """
        Returns the name of the current value
        :return:
        """
        if self.alias:
            return self.alias
        return self.final_name

    def get_value(self):
        """
        Returns the value of the object
        :return:
        """
        return self.value

    @staticmethod
    def get_other_value(other):
        """
        Return the appropriate value based on the type of other
        :param other:
        :return:
        """
        if isinstance(other, Value):
            return other.get_value()
        return other

    def set_alias(self, alias):
        """
        Sets the alias and final name for the value object
        :param alias:
        :return:
        """
        self.alias = alias
        self.final_name = alias

    def set_type(self, type_name: str):
        self.value = self.value.cast(type_name)

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.value > other.value
        return self.value > other

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.value < other.value
        return self.value < other

    def __ge__(self, other):
        if isinstance(other, Value):
            return self.value >= other.value
        return self.value >= other

    def __le__(self, other):
        if isinstance(other, Value):
            return self.value <= other.value
        return self.value <= other

    def __ne__(self, other):
        if isinstance(other, Value):
            return self.value != other.value
        return self.value != other

    def __or__(self, other):
        if isinstance(other, Value):
            return self.value | other.value
        return self.value | other

    def __and__(self, other):
        if isinstance(other, Value):
            return self.value & other.value
        return self.value & other


@dataclass
class Literal(Value):
    """
    Stores literal data
    """
    literal_count: ClassVar[int] = 0

    def __post_init__(self):
        super().__post_init__()
        if not self.alias:
            self.alias = f"_literal{self.literal_count}"
            type(self).literal_count += 1
        self.to_ibis_literal()

    def to_ibis_literal(self):
        if not isinstance(self.value, ValueExpr):
            self.value = ibis.literal(self.value)

    def __repr__(self):
        return Value.__repr__(self) + ")"

    @classmethod
    def reset_literal_count(cls):
        cls.literal_count = 0

@dataclass
class Number(Literal):
    """
    Stores numerical data
    """
    pass

@dataclass
class String(Literal):
    """
    Store information about a string literal
    """
    pass

@dataclass
class Date(Literal):
    """
    Store information about a date literal
    """
    pass

@dataclass
class Bool(Literal):
    """
    Store information about a date literal
    """
    pass


@dataclass
class DerivedColumn(Value):
    """
    Base class for expressions and aggregates
    """
    expression_count: ClassVar[int] = 0
    function: str = ""

    def __post_init__(self):
        if self.alias:
            self.final_name = self.alias
        else:
            if isinstance(self.value, (Series, Column)) or isinstance(
                self, (Aggregate, Expression)
            ):
                self.final_name = f"_col{self.expression_count}"
                self.alias = self.final_name
                DerivedColumn.increment_expression_count()
            else:
                self.final_name = str(self.value)

    def __repr__(self):
        display = Value.__repr__(self)
        if self.function:
            display += f", function={self.function}"
        return display + ")"

    @classmethod
    def increment_expression_count(cls):
        cls.expression_count += 1

    @classmethod
    def reset_expression_count(cls):
        cls.expression_count = 0


class Expression(DerivedColumn):
    """
    Store information about an sql_object
    """

    def __init__(self, value, alias="", typename="", function=""):
        DerivedColumn.__init__(self, value, alias, typename, function)

    def get_name(self) -> str:
        return self.alias


class Column(Value):
    """
    Store information about columns
    """

    def __init__(
        self, name: str, alias="", typename="", value: Optional[AnyColumn] = None
    ):
        Value.__init__(self, value, alias, typename)
        self.name = name
        if self.alias:
            self.final_name = self.alias
        else:
            self.final_name = self.name
        self._table: Optional[Table] = None

    def __repr__(self):
        display = Value.__repr__(self)
        display += f", name={self.name}"
        display += f", table={self._table}"
        return display + ")"

    def __eq__(self, other):
        other = self.get_other_value(other)
        return self.value == other

    def __gt__(self, other):
        other = self.get_other_value(other)
        return self.value > other

    def __lt__(self, other):
        other = self.get_other_value(other)
        return self.value < other

    def __ge__(self, other):
        other = self.get_other_value(other)
        return self.value >= other

    def __le__(self, other):
        other = self.get_other_value(other)
        return self.value <= other

    def get_table(self):
        return self._table

    def set_table(self, table: Table):
        self._table = table

    def desc(self):
        self.value = ibis.desc(self.value)
        return self


class CountStar(Column):
    def __init__(self):
        super().__init__("*", alias="*")


class Aggregate(DerivedColumn):
    """
    Store information about aggregations
    """

    def __init__(self, value: Union[AnyScalar, CountStar], alias="", typename=""):
        DerivedColumn.__init__(self, value, alias, typename)


class GroupByColumn(Column):
    def __init__(
        self,
        name: str,
        groupby_name: str,
        alias="",
        typename="",
        value: Optional[AnyColumn] = None,
    ):
        super().__init__(name, alias, typename, value)
        self.group_by_name = groupby_name

    @classmethod
    def from_column_type(cls, column: Column):
        return cls(
            name=column.name,
            groupby_name=column.name,
            alias=column.alias,
            typename=column.typename,
            value=column.value,
        )

    def set_ibis_name_to_name(self):
        self.value = self.value.name(self.get_name())


@dataclass
class Subquery(Table):
    """
    Wrapper for subqueries
    """

    def __post_init__(self, value: TableExpr):
        self._value = value
        self.alias = self.name


class JoinBase:
    def __init__(
        self, left_table: Table, right_table: Table, join_type: str,
    ):
        self.left_table: Table = left_table
        self.right_table: Table = right_table
        self.join_type: str = join_type

    def __repr__(self):
        return (
            f"{type(self).__name__}(left={self.left_table}, right="
            f"{self.right_table}, type={self.join_type})"
        )


class Join(JoinBase):
    """
    Wrapper for join related info
    """

    def __init__(
        self,
        left_table: Table,
        right_table: Table,
        join_type: str,
        left_on: str,
        right_on: str,
    ):
        super().__init__(left_table, right_table, join_type)
        self.left_on = left_on
        self.right_on = right_on


class CrossJoin(JoinBase):
    def __init__(
        self, left_table: Table, right_table: Table,
    ):
        super().__init__(left_table, right_table, "cross")
