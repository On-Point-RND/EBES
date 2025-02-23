from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, ArrayType
from sktime.datasets import load_from_tsfile_to_dataframe
import pandas as pd
import numpy as np


def prep_pdf(pdf: pd.DataFrame):
    return pdf.assign(
        dim_0=lambda df: df["dim_0"].apply(lambda s: s.tolist()),
        class_vals=lambda df: df["class_vals"].astype(int) - 1,
        time=lambda df: [np.arange(len(s)).tolist() for s in df["dim_0"]],
        _seq_len=96,
        _last_time=95,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing unzipped ts files",
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
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    for split in ("train", "test"):
        pdf = load_from_tsfile_to_dataframe(
            args.data_path / f"ElectricDevices_{split.upper()}.ts",
            return_separate_X_and_y=False,
        )
        pdf = prep_pdf(pdf)
        (
            spark.createDataFrame(pdf)
            .withColumn("dim_0", F.col("dim_0").cast(ArrayType(FloatType())))
            .withColumn("time", F.col("time").cast(ArrayType(FloatType())))
            .withColumn("_last_time", F.col("_last_time").cast(FloatType()))
            .withColumn("index", F.monotonically_increasing_id())
            .coalesce(1)
            .write.parquet((args.save_path / split).as_posix(), mode=mode)
        )


if __name__ == "__main__":
    main()
