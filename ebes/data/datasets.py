import math

from torch.utils.data import IterableDataset, get_worker_info
import pandas as pd
import numpy as np


def series(df: pd.DataFrame) -> list[pd.Series]:
    """Return list of DataFrame rows as a series."""

    return list(map(lambda idx_and_s: idx_and_s[1], df.iterrows()))


class SeriesDataset(IterableDataset):
    """An iterable dataset over the DataFrame rows."""

    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        query: str | None = None,
        drop_incomplete: bool = False,
        shuffle: bool = False,
        loop: bool = False,
        random_seed: int | None = None,
    ):
        """Initialize SeriesDataset.

        The dataset yields lists of pd.Series of length not greater than `batch_size`.
        The last incomplete batch can be omitted. The data is optionally shuffled before
        iterating. The dataset can be looped. In this case after all dataset is
        exhausted, the data is reshufled (if enabled) and the iteration continues
        without raising StopIteration, so the dataset becomes infinite. Incomplete batch
        still can occur in such case.

        The SeriesDataset can be safely used with PyTorch multiprocessing. To preserve
        the internal state (progress through the data in case of looped dataset or
        the state of the random generator) pass `persistent_workers=True` to the
        DataLoader.

        Args:
            data: a pandas DataFrame to iterate over.
            batch_size: number of rows yielded at a time. Must be > 0.
            query: optional query to filter the data befor loading.
            drop_incomplete: whether to drop or keep the last incomplete batch.
            shuffle: if True, the DataFrame is shuffled.
            loop: whether to loop the dataset.
            random_seed: seed to initialize the random generator for shuffling.
        """

        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        if query:
            data = data.query(query)

        self._data = data
        self._bs = batch_size
        self._drop_last = drop_incomplete
        self._shuffle = shuffle
        self._loop = loop
        self._gen = np.random.default_rng(random_seed)

        self._exhausted = True

    def _refresh(self):
        worker_info = get_worker_info()
        total_workers = 1
        worker_num = 0
        if worker_info is not None:
            total_workers = worker_info.num_workers
            worker_num = worker_info.id

        if self._shuffle:
            self._data = self._data.sample(frac=1, random_state=self._gen)

        to = len(self._data)
        if self._drop_last:
            to -= self._bs
        self._idx_iter = iter(
            range(worker_num * self._bs, to, self._bs * total_workers)
        )

        self._exhausted = False

    def __iter__(self):
        if self._exhausted:
            self._refresh()
        return self

    def __next__(self) -> list[pd.Series]:
        try:
            idx = next(self._idx_iter)
            return series(self._data.iloc[idx : idx + self._bs])
        except StopIteration:
            self._exhausted = True
            if not self._loop:
                raise
            self._refresh()
            return self.__next__()


class SizedSeriesDataset(SeriesDataset):
    """The same as SeriesDataset, but has __len__ method implemented."""

    def __len__(self):
        round_fn = math.floor if self._drop_last else math.ceil
        return round_fn(len(self._data) / self._bs)
