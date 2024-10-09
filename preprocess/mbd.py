from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from common import cat_freq, collect_lists


CAT_FEATURES = [
    "event_type",
    "event_subtype",
    "currency",
    "src_type11",
    "src_type12",
    "dst_type11",
    "dst_type12",
    "src_type21",
    "src_type22",
    "src_type31",
    "src_type32",
]
NUM_FEATURES = ["amount"]
INDEX_COLUMNS = [
    "client_id",
    "mon",
    "fold",
    "bcard_target",
    "cred_target",
    "zp_target",
    "acquiring_target",
]
ORDERING_COLUMNS = ["event_time"]
TEST_FOLD = 4


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
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df_trx = spark.read.parquet((args.data_path / "detail" / "trx").as_posix())
    df_target = spark.read.parquet((args.data_path / "targets").as_posix())
    df = (
        df_trx.withColumn("mon", F.last_day(F.date_add(F.last_day("event_time"), 1)))
        .join(df_target, on=["client_id", "mon", "fold"])
        .select(*INDEX_COLUMNS, *ORDERING_COLUMNS, *CAT_FEATURES, *NUM_FEATURES)
    )

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

    train_df = df.filter(f"fold != {TEST_FOLD}")
    test_df = df.filter(f"fold == {TEST_FOLD}")
    train_df.coalesce(1).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.coalesce(1).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
