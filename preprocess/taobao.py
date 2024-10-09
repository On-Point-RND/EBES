from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, TimestampType

from common import cat_freq, collect_lists


CAT_FEATURES = ["item_id", "behavior_type"]
INDEX_COLUMNS = ["user_id", "payment_next_7d"]
ORDERING_COLUMNS = ["time"]
TARGET_VALS = [0, 1]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path CSV train user",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--save-path",
        help="Where to save preprocessed parquets",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cat-codes-path",
        help="Path where to save codes for categorical features",
        type=Path,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df = spark.read.csv(args.data_path.as_posix(), header=True)
    df = df.select(
        F.col("user_id").cast(StringType()),
        F.col("item_id").cast(LongType()),
        F.col("behavior_type").cast(LongType()),
        F.col("time").cast(TimestampType()),
    )

    def extract_data(
        df: DataFrame,
        user_suffix: str,
        start_date: str,
        mid_date: str,
        end_date: str,
    ) -> DataFrame:
        w_hist = df.filter(
            f"time >= '{start_date}' and time < '{mid_date}'"
        ).withColumn("user_id", F.concat(F.col("user_id"), F.lit(user_suffix)))
        w_payment = (
            df.filter(
                f"time >= '{mid_date}' and time < '{end_date}' and behavior_type == 4"
            )
            .select("user_id")
            .distinct()
            .select(
                F.concat(F.col("user_id"), F.lit(user_suffix)).alias("user_id"),
                F.lit(1).alias("payment_next_7d"),
            )
        )
        return w_hist.join(w_payment, on="user_id", how="left").na.fill(
            0, subset=["payment_next_7d"]
        )

    df1 = extract_data(df, "_1", "2014-11-18", "2014-11-25", "2014-12-02")
    df2 = extract_data(df, "_2", "2014-11-25", "2014-12-02", "2014-12-09")
    df_train = df1.union(df2)
    df_test = extract_data(df, "_3", "2014-12-02", "2014-12-09", "2014-12-16")

    vcs = cat_freq(df_train, CAT_FEATURES)
    for vc in vcs:
        df_train = vc.encode(df_train)
        df_test = vc.encode(df_test)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    train_df = collect_lists(
        df_train, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS
    )
    test_df = collect_lists(df_test, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS)

    train_df.coalesce(1).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.coalesce(1).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
