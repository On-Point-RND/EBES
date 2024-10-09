from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyspark.sql import functions as F
from pyspark.sql import Window, SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import LongType


NA_VALUE = 0


# TODO: docs
class CatMap:
    def __init__(self, map_df: DataFrame):
        if (
            len(map_df.columns) != 2
            or map_df.columns[1] != "_code"
            or map_df.schema["_code"].dataType != LongType()
        ):
            raise ValueError(
                "DataFrame must contain exactly 2 columns. "
                "The 2nd one must be named `_code` and have type LongType"
            )

        self._df = map_df

    @property
    def feature_name(self):
        return self._df.columns[0]

    @property
    def df(self):
        return self._df

    def encode(self, df: DataFrame) -> DataFrame:
        return (
            df.join(self._df, on=self.feature_name, how="left")
            .drop(self.feature_name)
            .na.fill(NA_VALUE, subset=["_code"])
            .withColumnRenamed("_code", self.feature_name)
        )

    def decode(self, df: DataFrame) -> DataFrame:
        return (
            df.join(
                self._df.select(
                    F.col(self.feature_name).alias("_code"),
                    F.col("_code").alias(self.feature_name),
                ),
                on=self.feature_name,
                how="left",
            )
            .drop(self.feature_name)
            .withColumnRenamed("_code", self.feature_name)
        )

    def write(self, path: Path, mode: str | None = None):
        self._df.write.parquet(path.as_posix(), mode=mode)

    @classmethod
    def read(cls, path: Path):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore
        df = spark.read.parquet(path.as_posix())
        return cls(df)


def collect_lists(
    df: DataFrame,
    group_by: str | Iterable[str],
    order_by: str | Iterable[str],
) -> DataFrame:
    """Collect lists and add auxiliary columns.

    The function collect all sequence elements in the dataframe in lists grouping by
    the `group_by` columns and ordering by the `order_by` columns. It also computes the
    auziliary information: sequence lengths and the last value(s) in the `order_by`
    column(s). The latter columns are named as `order_by` columns with prefix "_last_",
    the column containing sequence lengths has name "_seq_len".

    Args:
        df: DataFrame containing all sequences.
        group_by: column(s) identifying a sequence.
        order_by: column(s) used for ordering sequences.

    Return:
        a dataframe with collected lists and auxiliary columns.
    """

    if isinstance(order_by, str):
        order_by = (order_by,)
    order_by = list(order_by)

    if isinstance(group_by, str):
        group_by = (group_by,)
    group_by = list(group_by)

    seq_cols = list(set(df.columns) - set(group_by) - set(order_by))

    return (
        df.select(*group_by, F.struct(*order_by, *seq_cols).alias("s"))
        .groupBy(*group_by)
        .agg(F.sort_array(F.collect_list("s")).alias("s"))
        .select(
            *group_by,
            *map(lambda c: "s." + c, order_by + seq_cols),
            F.size("s").alias("_seq_len"),
            *map(lambda c: F.element_at("s." + c, -1).alias("_last_" + c), order_by),
        )
    )


def cat_freq(df: DataFrame, cols: Iterable[str]) -> list[CatMap]:
    """Computes the value frequency ranks for columns.

    Counts occurencies of each value in col (excluding NULL and NaN values) and returns
    dataframes containing mappings from values in col to their frequency ranks (starting
    from 1 for the most frequent value). Each dataframe has 2 columns: 'col' with column
    values and '_code' with the frequency ranks.

    Args:
        df: dataframe.
        cols: columns for which to count the occurencies.

    Returns:
        list of dataframes with values frequency ranks for each column.
    """

    val_counts = []
    for col in cols:
        map_df = (
            df.select(col)
            .dropna(subset=[col])
            .groupBy(col)
            .count()
            .select(
                col,
                (
                    F.row_number().over(
                        # dummy partition F.lit(0) to suppress WindowExec warning
                        # "No Partition Defined for Window operation! ..."
                        Window.partitionBy(F.lit(0)).orderBy(F.col("count").desc())
                    )
                )
                .cast(LongType())
                .alias("_code"),
            )
        )
        val_counts.append(CatMap(map_df))

    return val_counts


def train_test_split(
    df: DataFrame,
    test_frac: float,
    index_col: str,
    stratify_col: str | None = None,
    stratify_col_vals: list[Any] | None = None,
    random_seed: int = 0,
) -> tuple[DataFrame, DataFrame]:
    """Split dataset on train and test parts.

    Stratified random splitting dataframe rows on train and test. It uses
    `pyspark.sql.DataFrame.sampleBy` internally with `col` parameter set to
    `stratify_col`.

    Args:
        df: dataframe to split.
        test_frac: test fraction (0 <= test_frac <= 1).
        index_col: a column with dataset items index to sample.
        stratify_col: a column to stratify by.
        stratify_col_vals: unique values of the stratification colums. Uf not set, the
            values are calculated using `pyspark.sql.DataFrame.distinct()`.
        random_seed: random seed for reproducibility.

    Returns:
        a tuple of train and test dataframes.
    """

    if stratify_col is None:
        index = df.select(index_col).coalesce(1).sort(index_col).cache()
        test_index = index.sample(fraction=test_frac, seed=random_seed).cache()
        test_df = df.join(test_index, on=index_col)
        train_index = index.select(index_col).subtract(test_index)
        train_df = df.join(train_index, on=index_col)
        return train_df, test_df

    fractions = None
    if stratify_col_vals is not None:
        fractions = {val: test_frac for val in stratify_col_vals}
    else:
        df.cache()
        fractions = (
            df.select(stratify_col)
            .distinct()
            .withColumn("_fraction", F.lit(test_frac))
            .rdd.collectAsMap()
        )

    index = df.select(index_col, stratify_col).coalesce(1).sort(index_col).cache()

    test_index = (
        index.sampleBy(stratify_col, fractions, random_seed).select(index_col).cache()
    )
    test_df = df.join(test_index, on=index_col)

    train_index = index.select(index_col).subtract(test_index)
    train_df = df.join(train_index, on=index_col)

    return train_df, test_df
