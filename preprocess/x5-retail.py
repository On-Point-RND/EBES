from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType, TimestampType
from pyspark.ml.feature import Bucketizer

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = [
    "store_id",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "segment_id",
    "brand_id",
    "vendor_id",
    "gender",
]
INDEX_COLUMNS = [
    "client_id",
    "age",
    "gender",
    "first_issue_date",
    "first_redeem_date",
]
ORDERING_COLUMNS = [
    "transaction_datetime",
]
AGE_BOUNDS = [10.0, 35.0, 45.0, 60.0, 90.0]
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
        "--train-partitions",
        help="Number of parquet partitions for train dataset",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--test-partitions",
        help="Number of parquet partitions for test dataset",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df_prods = (
        spark.read.csv((args.data_path / "products.csv").as_posix(), header=True)
        .withColumn("netto", F.col("netto").cast(FloatType()))
        .withColumn("is_own_trademark", F.col("is_own_trademark").cast(LongType()))
        .withColumn("is_alcohol", F.col("is_alcohol").cast(LongType()))
    )

    df_clients = (
        spark.read.csv((args.data_path / "clients.csv").as_posix(), header=True)
        .withColumn("first_issue_date", F.col("first_issue_date").cast(TimestampType()))
        .withColumn(
            "first_redeem_date", F.col("first_redeem_date").cast(TimestampType())
        )
        .withColumn("age", F.col("age").cast(FloatType()))
        .filter("age >= 10.0 and age <= 90.0")  # as in CoLES
    )

    df_tx = (
        spark.read.csv((args.data_path / "purchases.csv").as_posix(), header=True)
        .withColumn(
            "transaction_datetime", F.col("transaction_datetime").cast(TimestampType())
        )
        .withColumn(
            "regular_points_received",
            F.col("regular_points_received").cast(FloatType()),
        )
        .withColumn(
            "express_points_received",
            F.col("express_points_received").cast(FloatType()),
        )
        .withColumn(
            "regular_points_spent", F.col("regular_points_spent").cast(FloatType())
        )
        .withColumn(
            "express_points_spent", F.col("express_points_spent").cast(FloatType())
        )
        .withColumn("purchase_sum", F.col("purchase_sum").cast(FloatType()))
        .withColumn("product_quantity", F.col("product_quantity").cast(FloatType()))
        .withColumn("trn_sum_from_iss", F.col("trn_sum_from_iss").cast(FloatType()))
        .withColumn("trn_sum_from_red", F.col("trn_sum_from_red").cast(FloatType()))
    )

    mode = "overwrite" if args.overwrite else "error"

    df = df_tx.join(df_prods, on="product_id").join(df_clients, on="client_id")

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

    # split age on buckers as in CoLES
    df = (
        Bucketizer(
            splits=AGE_BOUNDS,
            inputCol="age",
            outputCol="age_clf",
            handleInvalid="error",
        )
        .transform(df)
        .withColumn("age_clf", F.col("age_clf").cast(LongType()))
        .cache()
    )

    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col="client_id",
        stratify_col="age_clf",
        stratify_col_vals=list(range(len(AGE_BOUNDS) - 1)),
        random_seed=args.split_seed,
    )

    train_df.repartition(args.train_partitions).write.parquet(
        (args.save_path / "train").as_posix(), mode=mode
    )
    test_df.repartition(args.test_partitions).write.parquet(
        (args.save_path / "test").as_posix(), mode=mode
    )


if __name__ == "__main__":
    main()
