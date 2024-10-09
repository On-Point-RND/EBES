"""The module contains classes that expose a pd.DataFrame interface to datasets."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class PandasDataAccessor(ABC):
    """Abstract class for all data accessors.

    Data accessor is responsible for splitting the data on train/test/whatever,
    filtering the data and exposing pd.DataFrame interface to it. The splits are
    configred in subclass `__init__` methods and accessed by their index. Each subclass
    splits the data (using any specified strategy) and returns a split by its positive
    index as a result of the `get_split` method.
    """

    @abstractmethod
    def get_split(self, split_idx: int) -> pd.DataFrame | Sequence[pd.DataFrame]:
        """Get split by its index.

        Args:
            split_idx: positive index of split.

        Returns:
            A dataframe or a sequence of dataframes with data in given split.
        """
        ...


class InMemoryPandasDataAccessor(PandasDataAccessor):
    """Data accessor that keeps all data in memory."""

    def __init__(
        self,
        *,
        parquet_path: str,
        split_sizes: Sequence[float],
        data_queries: Sequence[str] | None = None,
        split_by_col: str | list[str] | None = None,
        random_split: bool = False,
        split_seed: int | None = None,
    ):
        """Initialize InMemoryPandasDataAccessor.

        Provide path to data, split sizes, filtering queries and splitting strategy.
        You can split data randomly, sorting by column(s) or as it is stored if spliting
        parameters are not passed.

        Args:
            parquet_path: path to parquet file(s) containing preprocessed data.
            split_sizes: split sizes as a fraction of all data. The bounds are not
                checked. If the sum is less than 1, the last part is never accessed,
                if greater than 1, the last split will be of size less than specified or
                will be completely out of bounds and exception will be raised.
            data_queries: pandas query string for data filtering. To skip filtering a
                particular split, put empty string on the corresponding place in
                sequence. The data is filtering **after** splitting, so the size of
                filtered split will be less than specified.
            split_by_col: Sort data in the ascending order by `split_by_col` column(s)
                and split data sequentially. Mutually exclusive with `random_split`.
            random_split: Split data randomly. Mutually exclusive with `split_by_col`.
            split_seed: seed used for random data splitting. If None, global random
                state is used. Ignored if `split_by_col` is set.
        """

        if split_by_col is not None and random_split:
            raise ValueError(
                "You cat specify at most one of `split_by_col` and `random_split`"
            )

        logger.info("reading parquet file, it may take some time")
        self._data = pd.read_parquet(parquet_path)
        self._queries = data_queries

        if split_by_col is not None:
            self._data = self._data.sort_values(split_by_col)
        if random_split:
            self._data = self._data.sample(frac=1, random_state=split_seed)

        self._split_bounds = np.ceil(
            np.cumsum([0, *split_sizes]) * len(self._data)
        ).astype(int)

    def get_split(self, split_idx) -> pd.DataFrame:
        fr, to = self._split_bounds[[split_idx, split_idx + 1]]
        df = self._data.iloc[fr:to]

        if self._queries is not None and self._queries[split_idx]:
            df = df.query(self._queries[split_idx])

        return df
