import os
import shutil
import zipfile
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe

import requests
from sklearn.preprocessing import LabelEncoder

parser = ArgumentParser()
parser.add_argument(
    "--data-path",
    help="Path to directory containing ts files",
    required=True,
    type=Path,
)
parser.add_argument(
    "--save-path",
    help="Where to save preprocessed parquets",
    required=True,
    type=Path,
)

args = parser.parse_args()


def data_verifier(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    directories = [
        name
        for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    ]
    if directories:
        print(f"The {data_path} data is already existed")
    else:
        file_url = "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip"
        downloader(file_url, data_path)


def downloader(file_url, data_path):
    # Define the path to download
    path_to_download = data_path
    # Send a GET request to download the file
    response = requests.get(file_url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the downloaded file
        file_path = os.path.join(path_to_download, "Multivariate2018_ts.zip")
        with open(file_path, "wb") as file:
            # Track the progress of the download
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024 * 100  # 1KB
            downloaded_size = 0

            for data in response.iter_content(block_size):
                file.write(data)
                downloaded_size += len(data)

                # Calculate the download progress percentage
                progress = (downloaded_size / total_size) * 100

                # Print the progress message
                print(f" Download in progress: {progress:.2f}%")

        # Extract the contents of the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(path_to_download)

        # Remove the downloaded zip file
        os.remove(file_path)

        shutil.move(
            data_path / "Multivariate_ts/SpokenArabicDigits",
            data_path / "SpokenArabicDigits",
        )
        shutil.rmtree(data_path / "Multivariate_ts")

        print(f"{data_path} Datasets downloaded and extracted successfully.")
    else:
        print(f"Failed to download the {data_path} please update the file_url")
    return


def preprocess(data_path, save_path):
    # Load original data
    print("Loading and preprocessing data ...")
    train_file = data_path / "SpokenArabicDigits/SpokenArabicDigits_TRAIN.ts"
    test_file = data_path / "SpokenArabicDigits/SpokenArabicDigits_TEST.ts"

    train_df, y_train = load_from_tsfile_to_dataframe(train_file)
    test_df, y_test = load_from_tsfile_to_dataframe(test_file)

    # Correct type
    train_df = train_df.map(lambda x: np.array(x, dtype=np.float32))
    test_df = test_df.map(lambda x: np.array(x))

    # Add sequence length column
    train_df["_seq_len"] = train_df.iloc[:, 0].map(len)
    test_df["_seq_len"] = test_df.iloc[:, 0].map(len)

    # Determine the maximum sequence length
    train_max_seq_len = int(np.max(train_df["_seq_len"]))
    test_max_seq_len = int(np.max(test_df["_seq_len"]))
    max_seq_len = np.max([train_max_seq_len, test_max_seq_len])

    # Create the 'time' column as pd.Series
    train_df["time"] = (
        train_df["_seq_len"].map(lambda x: np.arange(x, dtype=np.float32)) / max_seq_len
    )
    test_df["time"] = (
        test_df["_seq_len"].map(lambda x: np.arange(x, dtype=np.float32)) / max_seq_len
    )

    # Add sequence index column
    train_df["seq_id"] = train_df.index
    test_df["seq_id"] = test_df.index

    # Encode and add target column
    label_encoder = LabelEncoder()
    combined_labels = np.concatenate([y_train, y_test])
    label_encoder.fit(combined_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    train_df["target"] = y_train_encoded
    test_df["target"] = y_test_encoded

    print(f"{len(y_train_encoded)} samples will be used for training")
    print(f"{len(y_test_encoded)} samples will be used for testing")

    # Save the DataFrames as Parquet files
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_save_path = save_path / "train"
    test_save_path = save_path / "test"
    train_df.to_parquet(train_save_path, index=False)
    test_df.to_parquet(test_save_path, index=False)

    print(f"Training data saved to {train_save_path}")
    print(f"Testing data saved to {test_save_path}")


if __name__ == "__main__":
    data_verifier(args.data_path)
    preprocess(args.data_path, args.save_path)
