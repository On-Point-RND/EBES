from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType, StringType, TimestampType
import pandas as pd

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = ["CRR"]
NUM_FEATURES = [
    "Temp",
    "SpO2",
    "HR",
    "RR",
    "SBP",
    "DBP",
    "TGCS",
    "FiO2",
    "Glucose",
    "pH",
]
INDEX_COLUMNS = ["hadm_id", "length_of_stay", "hospital_expire_flag"]
TEST_FRACTION = 0.2

# from https://github.com/mlds-lab/interp-net/blob/master/src/mimic_data_extraction.py
ITEMID_FEATS = """
SpO2 - 646, 220277
HR - 211, 220045
RR - 618, 615, 220210, 224690
SBP - 51,442,455,6701,220179,220050
DBP - 8368,8440,8441,8555,220180,220051
Temp(F) - 223761,678
Temp(C) - 223762,676
TGCS - 198, 226755, 227013
CRR - 3348, 115, 223951, 8377, 224308
FiO2 - 2981, 3420, 3422, 223835
Glucose - 807,811,1529,3745,3744,225664,220621,226537
pH - 780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243
"""


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing gzipped CSV files",
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
        "--split-seed",
        help="Random seed used to split the data on train and test",
        default=0,
        type=int,
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

    df_adm = spark.read.csv(
        (args.data_path / "ADMISSIONS.csv.gz").as_posix(), header=True
    ).select(
        F.col("HADM_ID").cast(LongType()).alias("hadm_id"),
        (
            (
                F.col("DISCHTIME").cast(TimestampType())
                - F.col("ADMITTIME").cast(TimestampType())
            ).cast(LongType())
            / 3600
        ).alias("length_of_stay"),
        F.col("HOSPITAL_EXPIRE_FLAG").cast(LongType()).alias("hospital_expire_flag"),
    )

    data, feature_names = [], []
    for row in ITEMID_FEATS.strip().split("\n"):
        feature_name, ids_str = row.split(" - ")
        feature_names.append(feature_name)
        ids = list(map(int, ids_str.split(",")))
        data.append([feature_name, ids])

    df_features = spark.createDataFrame(
        pd.DataFrame(
            columns=["feature_name", "itemid"],  # type: ignore
            data=data,
        ).explode("itemid", ignore_index=True)
    )

    def within(
        col,
        lower: float | None = None,
        upper: float | None = None,
        clip: bool = False,
    ):
        lv, uv = None, None
        if clip:
            lv, uv = lower, upper

        if lower is not None:
            col = F.when(col < lower, lv).otherwise(col)
        if upper is not None:
            col = F.when(col > upper, uv).otherwise(col)
        return col

    df_values = (
        spark.read.csv((args.data_path / "CHARTEVENTS.csv.gz").as_posix(), header=True)
        .select(
            F.col("HADM_ID").cast(LongType()).alias("hadm_id"),
            F.col("CHARTTIME").cast(TimestampType()).alias("charttime"),
            F.col("ITEMID").cast(LongType()).alias("itemid"),
            F.col("VALUE").alias("value"),
        )
        .join(df_features, on="itemid")
        .drop("itemid")
        .groupBy("hadm_id", "charttime")
        .pivot("feature_name", feature_names)
        .agg(F.any_value("value", ignoreNulls=True))
        .withColumn("SpO2", within(F.col("SpO2").cast(FloatType()), 0, 100))
        .withColumn("HR", within(F.col("HR").cast(FloatType()), 0, 220))
        .withColumn("RR", within(F.col("RR").cast(FloatType()), 0, 70))
        .withColumn("SBP", within(F.col("SBP").cast(FloatType()), 0, 250))
        .withColumn("DBP", within(F.col("DBP").cast(FloatType()), 0, 210))
        .withColumn("TGCS", within(F.col("TGCS").cast(LongType()), 3, 15))
        .withColumn("FiO2", within(F.col("FiO2").cast(FloatType()), 0, 100))
        .withColumn(
            "Glucose",
            within(
                F.trim(F.replace(F.col("Glucose"), F.lit("cs"), F.lit(""))).cast(
                    FloatType()
                ),
                1,
                1000,
            ),
        )
        .withColumn("pH", within(F.col("pH").cast(FloatType()), 5, 10))
        .withColumn("CRR", F.col("CRR").cast(StringType()))
        .withColumn(
            "CRR", F.replace(F.col("CRR"), F.lit("Normal <3 Seconds"), F.lit("Brisk"))
        )
        .withColumn(
            "CRR", F.replace(F.col("CRR"), F.lit("Normal <3 secs"), F.lit("Brisk"))
        )
        .withColumn(
            "CRR",
            F.replace(F.col("CRR"), F.lit("Abnormal >3 Seconds"), F.lit("Delayed")),
        )
        .withColumn(
            "CRR", F.replace(F.col("CRR"), F.lit("Abnormal >3 secs"), F.lit("Delayed"))
        )
        .withColumn("Temp", within(F.col("Temp(F)").cast(FloatType()), 15, 150))
        .withColumn("Temp(C)", within(F.col("Temp(C)").cast(FloatType()), 15, 150))
        .withColumn(
            "Temp",
            F.when(
                F.col("Temp") < 70,  # erroneous Farenheit
                F.col("Temp") * 1.8 + 32,
            ).otherwise(F.col("Temp")),
        )
        .withColumn(
            "Temp",
            F.when(
                F.col("Temp").isNull() & (F.col("Temp(C)") > 60),  # erroneous Celsius
                F.col("Temp(C)"),
            ).otherwise(F.col("Temp")),
        )
        .withColumn(
            "Temp",
            F.when(
                F.col("Temp").isNull() & (F.col("Temp(C)") < 60),  # correct Celsius
                F.col("Temp(C)") * 1.8 + 32,
            ).otherwise(F.col("Temp")),
        )
        .drop("Temp(F)", "Temp(C)")
    )

    df = df_adm.join(df_values, on="hadm_id")

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = (
        collect_lists(df, group_by=INDEX_COLUMNS, order_by="charttime")
        .withColumn("first_t", F.get("charttime", 0))
        .withColumn(
            "hours_since_adm",
            F.filter(
                F.transform(
                    "charttime",
                    lambda x: (x - F.col("first_t")).cast(LongType()) / 3600,
                ),
                lambda x: x <= 48,
            ),
        )
        .drop("charttime")
        .withColumn("_seq_len", F.size("hours_since_adm"))
        .withColumn("_last_hours_since_adm", F.element_at("hours_since_adm", -1))
    )
    for c in NUM_FEATURES + CAT_FEATURES:
        df = df.withColumn(c, F.slice(c, F.lit(1), F.col("_seq_len")))

    stratify_col = "hospital_expire_flag"
    stratify_col_vals = [0, 1]

    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col="hadm_id",
        stratify_col=stratify_col,
        stratify_col_vals=stratify_col_vals,
        random_seed=args.split_seed,
    )

    train_df.coalesce(1).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.coalesce(1).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
