from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pm4py
from tqdm.auto import tqdm
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, LongType

from common import cat_freq, collect_lists


tqdm.pandas()


CAT_FEATURES = [
    "Action",
    "org_resource",
    "concept_name",
    "EventOrigin",
    "lifecycle_transition",
    "case_LoanGoal",
    "case_ApplicationType",
]
NUM_FEATURES = [
    "case_RequestedAmount",
    "FirstWithdrawalAmount",
    "NumberOfTerms",
    "MonthlyCost",
    "CreditScore",
    "OfferedAmount",
]
INDEX_COLUMNS = ["OfferID", "Accepted"]
ORDERING_COLUMNS = ["time_timestamp"]
FIRST_TEST_DATE = "2016-10-20"


def get_seqs_from_app(df):
    seqs = []
    cr_off = df.query("`concept:name` == 'O_Create Offer'")
    for idx in cr_off.index:
        off = df.loc[:idx].copy()
        offer_id = df.at[idx + 1, "OfferID"]
        acc = df.at[idx, "Accepted"]
        off["OfferID"] = offer_id
        off["Accepted"] = acc
        seqs.append(off.drop(["Selected", "EventID"], axis=1))

    return pd.concat(seqs, axis=0)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing .xes.gz file",
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

    pdf = pm4py.read_xes((args.data_path / "BPI Challenge 2017.xes.gz").as_posix())
    pdf = pdf.groupby("case:concept:name", group_keys=False).progress_apply(
        get_seqs_from_app, include_groups=False
    )
    pdf.columns = map(lambda s: s.replace(":", "_"), pdf.columns)

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df = spark.createDataFrame(pdf)
    df = df.withColumn("Accepted", F.col("Accepted").cast(LongType()))
    for nc in NUM_FEATURES:
        df = df.withColumn(nc, F.col(nc).cast(FloatType()))

    last = (
        df.groupby("OfferID").agg(F.max("time_timestamp").alias("last_ev_dt")).cache()
    )
    train_clients = last.filter(f"last_ev_dt < '{FIRST_TEST_DATE}'").select("OfferID")
    test_clients = last.select("OfferID").subtract(train_clients)

    train_df = df.join(train_clients, on="OfferID")
    test_df = df.join(test_clients, on="OfferID")

    vcs = cat_freq(train_df, CAT_FEATURES)
    for vc in vcs:
        train_df = vc.encode(train_df)
        test_df = vc.encode(test_df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    train_df = collect_lists(
        train_df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )
    test_df = collect_lists(
        test_df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    train_df.coalesce(1).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.coalesce(1).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
