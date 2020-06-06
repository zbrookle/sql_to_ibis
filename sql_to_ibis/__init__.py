"""
isort:skip_file
"""

# flake8: noqa
from sql_to_ibis.sql_select_query import query, register_temp_table, remove_temp_table

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
