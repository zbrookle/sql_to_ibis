from typing import Set

AVG_AGGREGATIONS: Set[str] = {"avg", "mean"}
SUM_AGGREGATIONS: Set[str] = {"sum"}
NUMERIC_AGGREGATIONS: Set[str] = AVG_AGGREGATIONS.copy()
NUMERIC_AGGREGATIONS.update(SUM_AGGREGATIONS)
MIN_AGGREGATIONS: Set[str] = {"min", "minimum"}
MAX_AGGREGATIONS: Set[str] = {"max", "maximum"}
COUNT_AGGREGATIONS: Set[str] = {"count"}
COUNT_DISTINCT_AGGREGATIONS: Set[str] = {"countdistinct"}
