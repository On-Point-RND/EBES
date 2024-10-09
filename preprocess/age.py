from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = ["small_group"]
NUM_FEATURES = ["amount_rur"]
INDEX_COLUMNS = ["client_id", "bins"]
ORDERING_COLUMNS = ["trans_date"]
TARGET_VALS = [0, 1, 2, 3]
TEST_FRACTION = 0.2


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing CSV files",
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
        "--which-split",
        help="Whether to preprocess train set, test set or their union",
        choices=["train", "test", "union"],
        required=True,
    )
    parser.add_argument(
        "--cat-codes-path",
        help="Path where to save codes for categorical features",
        type=Path,
    )
    parser.add_argument(
        "--split-seed",
        help="Random seed used to split the data on train and test",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    df, df_kag_train, df_kag_test = None, None, None

    if args.which_split in ("train", "union"):
        df_kag_train = spark.read.csv(
            (args.data_path / "transactions_train.csv").as_posix(), header=True
        )
        df_kag_train = df_kag_train.select(
            F.col("client_id").cast(LongType()),
            F.col("trans_date").cast(LongType()),
            F.col("small_group").cast(LongType()),
            F.col("amount_rur").cast(FloatType()),
        )

        df_label = spark.read.csv(
            (args.data_path / "train_target.csv").as_posix(), header=True
        ).select(F.col("client_id").cast(LongType()), F.col("bins").cast(LongType()))

        df_kag_train = df_kag_train.join(df_label, on="client_id")

    if args.which_split in ("test", "union"):
        df_kag_test = spark.read.csv(
            (args.data_path / "transactions_test.csv").as_posix(), header=True
        )

        df_kag_test = df_kag_test.select(
            F.col("client_id").cast(LongType()),
            F.col("trans_date").cast(LongType()),
            F.col("small_group").cast(LongType()),
            F.col("amount_rur").cast(FloatType()),
        )

    if df_kag_train is not None and df_kag_test is not None:
        df_kag_test = df_kag_test.withColumn("bins", F.lit(None).cast(LongType()))
        df = df_kag_train.union(df_kag_test)
    elif df_kag_train is not None:
        df = df_kag_train
    elif df_kag_test is not None:
        df = df_kag_test
    else:
        raise ValueError("Something went wrong, train and test are None")

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    stratify_col, stratify_col_vals = None, None
    if df_kag_train is not None:  # target has non-null values
        stratify_col = "bins"
        stratify_col_vals = TARGET_VALS

    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col="client_id",
        stratify_col=stratify_col,
        stratify_col_vals=stratify_col_vals,
        random_seed=args.split_seed,
    )

    train_df.coalesce(1).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.coalesce(1).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
