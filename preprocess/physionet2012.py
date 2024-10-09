from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType, StringType

from common import cat_freq, collect_lists

CAT_FEATURES = [
    "Gender",
    "ICUType",
    "MechVent",
]
INDEX_COLUMNS = [
    "RecordID",
]
ORDERING_COLUMNS = [
    "Time",
]
N_RECORDS = 4000


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to unpacked zip with Physionet 2012 data",
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

    data_dir = Path(args.data_path)

    data_params = {
        "train": {
            "rec_dir": "set-a",
            "target": "Outcomes-a.txt",
        },
        "test": {
            "rec_dir": "set-b",
            "target": "Outcomes-b.txt",
        },
    }
    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    for split, params in data_params.items():
        pdfs = []
        for f in tqdm(data_dir.joinpath(params["rec_dir"]).iterdir(), total=N_RECORDS):
            pdfs.append(
                pd.read_csv(f, header=0).drop(0, axis=0).assign(RecordID=int(f.stem))
            )

        pdf = (
            pd.concat(pdfs, ignore_index=True, axis=0)
            .drop_duplicates(subset=["Time", "Parameter", "RecordID"], keep="last")
            .pivot(columns="Parameter", index=["RecordID", "Time"], values="Value")
            .reset_index()
        )
        hm = pdf.Time.str.split(":", n=2, expand=True).astype(int)
        pdf.Time = hm[0] + hm[1] / 60

        df = spark.createDataFrame(pdf)
        num_cols = list(set(df.columns) - set(CAT_FEATURES) - {"Time", "RecordID"})
        cast = (
            [F.col("RecordID").cast(LongType()), F.col("Time").cast(FloatType())]
            + [F.col(c).cast(FloatType()) for c in num_cols]
            + [F.col(c).cast(StringType()) for c in CAT_FEATURES]
        )
        df = df.select(*cast)

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

        target = spark.read.csv(
            args.data_path.joinpath(params["target"]).as_posix(), header=True
        )
        target = target.select(*[F.col(c).cast(LongType()) for c in target.columns])

        df = df.join(target, on="RecordID")
        df.coalesce(1).write.parquet((args.save_path / split).as_posix(), mode=mode)


if __name__ == "__main__":
    main()
