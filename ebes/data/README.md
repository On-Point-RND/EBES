## How to Preprocess Data

All data should be preprocessed before running any models. See `es-bench/preprocess` for examples.

### Rules:
1. The dataset should be saved in Parquet format.
2. Each row in the Parquet file corresponds to one sequence. The following columns are necessary:
    - Sequence ID
    - Target containing a single value for the sequence
    - "_seq_len" field
    - Each numerical feature should be collected as a NumPy array of **Float type** for the corresponding sequence. NaNs are acceptable.
    - Each categorical feature should be collected as a NumPy array of **Long type** for the corresponding sequence.
3. Categorical columns need to be encoded.
    - NaNs should be encoded as 0.
    - Other values should be encoded according to their frequency, i.e., the most popular category encoded as 1, the second most popular encoded as 2, etc.
