"""Batch transforms for data loading pipelines."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from dataclasses import dataclass
import random
import logging

import torch
import numpy as np

from ..types import Batch


logger = logging.getLogger(__name__)

MISSING_CAT_VAL = 0


class BatchTransform(ABC):
    """Base class for all batch transforms.

    The BatchTransform is a Callable object that modifies Batch in-place.
    """

    @abstractmethod
    def __call__(self, batch: Batch):
        """Apply transform to the batch."""
        ...


class UnsqueezeTarget(BatchTransform):
    """Unsqueeze last dimension in target array.

    Last linear layer for regression task produces tensors of shape (bs, 1). When
    calling MSE loss with target of shape (bs,), PyTorch expands it to the shape
    (bs, bs) and loss is computed incorrectly. This batch transform reshapes the target
    to (bs, 1), so MSE loss is computed correctly.
    """

    def __call__(self, batch: Batch):
        assert batch.target is not None
        batch.target = batch.target[..., None]


@dataclass
class RandomEventsPermutation(BatchTransform):
    """Permute events in sequence randomly.

    Time, target and masks are left unchanged.
    """

    keep_last: bool = False
    """If ``True`` the last event remains on its place, other are permuted."""

    def __call__(self, batch: Batch):

        max_len = batch.time.shape[0]
        bs = len(batch)
        i_len = torch.arange(max_len)[:, None]
        i_batch = torch.arange(bs)

        if self.keep_last:
            perm_len = torch.maximum(batch.lengths - 1, torch.tensor(1))
        else:
            perm_len = batch.lengths

        valid = (i_len < perm_len).float()
        permutation_within_len = torch.multinomial(valid.T, max_len).T

        if self.keep_last:
            permutation_within_len[batch.lengths - 1, i_batch] = batch.lengths - 1

        permutation_valid_padding = torch.where(
            i_len < batch.lengths,
            permutation_within_len,
            permutation_within_len[batch.lengths - 1, i_batch],
        )

        if batch.cat_features is not None:
            batch.cat_features = batch.cat_features[permutation_valid_padding, i_batch]

        if batch.num_features is not None:
            batch.num_features = batch.num_features[permutation_valid_padding, i_batch]

        if batch.cat_mask is not None:
            batch.cat_mask = batch.cat_mask[permutation_valid_padding, i_batch]

        if batch.num_mask is not None:
            batch.num_mask = batch.num_mask[permutation_valid_padding, i_batch]


class RandomTime(BatchTransform):
    """Replace time with uniformly disributed values."""

    def __call__(self, batch: Batch):
        assert isinstance(batch.time, torch.Tensor)
        batch.time = torch.rand_like(batch.time).sort(0).values


@dataclass
class RescaleTime(BatchTransform):
    """Rescale time: subtract location and divide by scale."""

    loc: float
    """Location to subtract from time."""
    scale: float
    """Scale to divide time by."""

    def __call__(self, batch: Batch):
        assert isinstance(batch.time, torch.Tensor)
        batch.time = batch.time.float()
        batch.time.sub_(self.loc).div_(self.scale)


@dataclass
class TimeToFeatures(BatchTransform):
    """Add time to numerical features.

    To apply this transform first cast time to Tensor.
    Has to be applied BEFORE mask creation. And AFTER DatetoTime
    """

    process_type: Literal["cat", "diff", "none"] = "none"
    """
    How to add time to features. The options are:

    - ``"cat"`` --- add absolute time to other numerical features,
    - ``"diff"`` --- add time intervals between sequential events. In this case the
      first interval in a sequence equals zero.
    - ``"none"`` --- do not add time to features. This option is added for the ease of
      optuna usage.
    """
    time_name: str = "time"
    """Name of new feature with time, default ``"time"``."""

    def __call__(self, batch: Batch):
        assert self.process_type in [
            "diff",
            "cat",
            "none",
        ], "time_process may only be cat|diff|none"
        assert isinstance(batch.time, torch.Tensor)
        if self.process_type == "none":
            return
        t = batch.time[..., None].clone()
        if self.process_type == "diff":
            t = t.diff(dim=0, prepend=t[[0]])

        if batch.num_features_names is None:
            batch.num_features_names = [self.time_name]
            assert batch.num_features is None
            batch.num_features = t
            return

        assert batch.num_features is not None
        batch.num_features_names.append(self.time_name)
        batch.num_features = torch.cat((batch.num_features, t), dim=2)


@dataclass
class DatetimeToFloat(BatchTransform):
    """Cast time from np.datetime64 to float by rescale.
    scale:
    """

    loc: str | np.datetime64
    """
    Location to subtract. If string is passed, it is converted to ``np.datetime64``
    beforehand.
    """
    scale: tuple[int, str] | np.timedelta64
    """
    Scale to divide time by. If tuple is passed, it is passed to the ``np.timedelta64``
    function. The first item is a value and the second is a unit.
    """

    def __post_init__(self):
        if isinstance(self.loc, str):
            self.loc = np.datetime64(self.loc)
        if isinstance(self.scale, Sequence):
            self.scale = np.timedelta64(*self.scale)

    def __call__(self, batch: Batch):
        assert isinstance(batch.time, np.ndarray)
        assert isinstance(self.loc, np.datetime64)
        assert isinstance(self.scale, np.timedelta64)
        batch.time = torch.asarray(
            (batch.time - self.loc) / self.scale, dtype=torch.float32
        )


@dataclass
class Logarithm(BatchTransform):
    """Apply natural logarithm to specific feature."""

    names: list[str]
    """Feature names to transform by taking the logarithm."""

    def __call__(self, batch: Batch):
        for name in self.names:
            batch[name] = torch.log1p(torch.abs(batch[name])) * torch.sign(batch[name])


@dataclass
class Rescale(BatchTransform):
    """Rescale feature: subtract location and divide by scale."""

    name: str
    """Feature name."""
    loc: Any
    """Value to subtract from the feature values."""
    scale: Any
    """Value to divide by the feature values."""

    def __call__(self, batch: Batch):
        batch[self.name].sub_(self.loc).div_(self.scale)


@dataclass
class ForwardFillNans(BatchTransform):
    """Fill NaN values by propagating forwad lase non-nan values.

    The algoritm starts from the second step. If some values are NaNs, the values from
    the prevoius step are used to fill them. If the first time step contains NaNs, some
    NaNs will not be filled after the forward pass. To handle it ``backward=True`` might
    be specified to fill remaining NaN values from last to first after the forwad pass.
    But even after a backward pass the batch may contain NaNs, if some feature has all
    NaN values. To fill it use ``FillNans`` transform.
    """

    backward: bool = False
    """Wether to do backward fill after the forwad fill (see the class description)."""

    def __call__(self, batch: Batch):
        if batch.num_features is None:
            return
        if batch.num_features.shape[0] == 1:
            return

        for i in range(1, batch.num_features.shape[0]):
            batch.num_features[i] = torch.where(
                torch.isnan(batch.num_features[i]),
                batch.num_features[i - 1],
                batch.num_features[i],
            )

        if not self.backward:
            return

        for i in range(batch.num_features.shape[0] - 2, -1, -1):
            batch.num_features[i] = torch.where(
                torch.isnan(batch.num_features[i]),
                batch.num_features[i + 1],
                batch.num_features[i],
            )


@dataclass
class FillNans(BatchTransform):
    """Fill NaNs with specified values."""

    fill_value: Mapping[str, float] | float
    """
    If float, all NaNs in all numerical features will be replaced with the
    ``fill_value``. Mapping sets feature-specific replacement values.
    """

    def __call__(self, batch: Batch):
        if batch.num_features is None:
            return

        if isinstance(self.fill_value, float | int):
            batch.num_features.nan_to_num_(nan=self.fill_value)
            return

        for name, val in self.fill_value.items():
            batch[name].nan_to_num_(nan=val)


class ContrastiveTarget(BatchTransform):
    """Set target for contrastive losses.

    New target is LongTensor such that items with different indices have different
    target labels.
    """

    def __call__(self, batch: Batch):
        if batch.index is None:
            raise ValueError("Batch must contain index")

        index = (
            batch.index
            if isinstance(batch.index, np.ndarray)
            else batch.index.cpu().numpy()
        )
        idx_map = {idx: i for i, idx in enumerate(np.unique(index))}
        batch.target = torch.tensor([idx_map[idx] for idx in index])


class TargetToLong(BatchTransform):
    """
    Cast target to LongTensor
    """

    def __call__(self, batch: Batch):
        if batch.target is not None:
            batch.target = batch.target.long()


@dataclass
class RandomSlices(BatchTransform):
    """Sample random slices from input sequences.

    The transform is taken from https://github.com/dllllb/coles-paper. It samples random
    slices from initial sequences. The batch size after this transform will be
    ``split_count`` times larger.
    """

    split_count: int
    """How many sample slices to draw for each input sequence."""
    cnt_min: int
    """Minimal sample sequence length."""
    cnt_max: int
    """Maximal sample sequence length."""
    short_seq_crop_rate: float = 1.0
    """
    Must be from (0, 1]. If ``short_seq_crop_rate`` < 1, and if a
        sequence of length less than cnt_min is encountered, the mininum sample
        length for this sequence is set as a ``short_seq_crop_rate`` time the actual
        sequence length.
    """
    seed: int | None = None
    """Value to seed the random generator."""

    def __post_init__(self):
        self._gen = np.random.default_rng(self.seed)

    def __call__(self, batch: Batch):

        lens = []
        times = []
        nums = []
        cats = []
        inds = []
        targets = []
        max_len = 0

        def add_slice(i, start, length):
            assert length > 0
            end = start + length
            lens.append(length)
            times.append(batch.time[start:end, i])
            inds.append(batch.index[i])
            if batch.num_features is not None:
                nums.append(batch.num_features[start:end, i])
            if batch.cat_features is not None:
                cats.append(batch.cat_features[start:end, i])
            if batch.target is not None:
                targets.append(batch.target[i])

        for i in range(len(batch)):
            c_len = batch.lengths[i].item()
            assert isinstance(c_len, int)
            if c_len < self.cnt_min and self.short_seq_crop_rate >= 1.0:
                for _ in range(self.split_count):
                    add_slice(i, 0, c_len)
                continue

            cnt_max = min(self.cnt_max, c_len)
            cnt_min = self.cnt_min
            if (
                int(c_len * self.short_seq_crop_rate) <= self.cnt_min
                and self.short_seq_crop_rate < 1.0
            ):
                cnt_min = max(int(c_len * self.short_seq_crop_rate), 1)

            if cnt_max > cnt_min:
                new_len = self._gen.integers(cnt_min, cnt_max, size=self.split_count)
            else:
                new_len = np.full(self.split_count, cnt_min)

            max_len = max(max_len, *new_len)
            available_start_pos = (c_len - new_len).clip(0, None)
            start_pos = (
                self._gen.uniform(size=self.split_count)
                * (available_start_pos + 1 - 1e-9)
            ).astype(int)

            for sp, ln in zip(start_pos, new_len):
                add_slice(i, sp, ln)

        def cat_pad(tensors, dtype):
            t0 = tensors[0]
            res = torch.zeros(max_len, len(tensors), *t0.shape[1:], dtype=dtype)
            for i, ten in enumerate(tensors):
                res[: ten.shape[0], i] = ten
                res[ten.shape[0] :, i] = ten[-1]
            return res

        batch.lengths = torch.tensor(lens)
        if batch.target is not None:
            batch.target = torch.tensor(targets, dtype=batch.target.dtype)
        if isinstance(batch.index, torch.Tensor):
            batch.index = torch.tensor(inds, dtype=batch.index.dtype)
        else:  # np.ndarray
            batch.index = np.array(inds, dtype=batch.index.dtype)

        batch.time = cat_pad(times, batch.time.dtype)
        if batch.cat_features is not None:
            batch.cat_features = cat_pad(cats, batch.cat_features.dtype)
        if batch.num_features is not None:
            batch.num_features = cat_pad(nums, batch.num_features.dtype)


@dataclass
class PrimeNetSampler(BatchTransform):
    """
    Contrastive sampling according to PrimeNet.

    Input:
        batch: Batch. Masks required.

    batch.num_features (T, B, D) -> (T, 2B, D)
    batch.cat_features (T, B, D) -> (T, 2B, D)

    Masks have additional dim for constrastive and interpolation:
    batch.num_mask (T, B, D) - > (T, 2B, D, 2)
    batch.cat_mask (T, B, D) - > (T, 2B, D, 2)
    """

    # max_len: int
    len_sampling_bound = [0.3, 0.7]
    dense_sampling_bound = [0.4, 0.6]
    mask_ratio_per_seg: float = 0.05
    segment_num: int = 3
    pretrain_tasks: str = "full2"

    def __call__(self, batch: Batch):
        lens = []
        times = []
        nums = []
        cats = []
        num_masks = []
        cat_masks = []
        inds = []
        targets = []
        max_len = 0

        def add_slice(idx, selected):
            assert isinstance(selected, set)
            selected = list(selected)
            lens.append(len(selected))
            times.append(batch.time[selected, idx])
            inds.append(batch.index[idx])
            if batch.num_features is not None:
                nums.append(batch.num_features[selected, idx])
            if batch.cat_features is not None:
                cats.append(batch.cat_features[selected, idx])
            # if batch.target is not None:
            #     targets.append(batch.target[idx])
            if batch.num_mask is not None:
                num_mask = batch.num_mask[selected, idx]
                if self.pretrain_tasks == "full2":
                    num_mask = self._seg_masking(mask=num_mask, timestamps=times[-1])
                num_masks.append(num_mask)
            if batch.cat_mask is not None:
                cat_mask = batch.cat_mask[selected, idx]
                if self.pretrain_tasks == "full2":
                    cat_mask = self._seg_masking(mask=cat_mask, timestamps=times[-1])
                cat_masks.append(cat_mask)

        for idx in range(len(batch)):
            selected_indices = self._time_sensitive_cl(
                batch.time[:, idx], batch.lengths[idx]
            )
            add_slice(idx, selected_indices)
            unselected_indices = set(range(batch.lengths[idx])) - selected_indices
            add_slice(idx, unselected_indices)

        max_len = max(max_len, *lens)

        def cat_pad(tensors, dtype):
            t0 = tensors[0]
            res = torch.zeros(max_len, len(tensors), *t0.shape[1:], dtype=dtype)
            for i, ten in enumerate(tensors):
                res[: ten.shape[0], i] = ten
            return res

        batch.lengths = torch.tensor(lens)
        batch.time = cat_pad(times, batch.time.dtype)
        if isinstance(batch.index, torch.Tensor):
            batch.index = torch.tensor(inds, dtype=batch.index.dtype)
        else:  # np.ndarray
            batch.index = np.array(inds, dtype=batch.index.dtype)
        if batch.cat_features is not None:
            batch.cat_features = cat_pad(cats, batch.cat_features.dtype)
        if batch.num_features is not None:
            batch.num_features = cat_pad(nums, batch.num_features.dtype)
        if batch.target is not None:
            # batch.target = torch.tensor(targets, dtype=batch.target.dtype)
            batch.target = None
        if batch.cat_mask is not None:
            batch.cat_mask = cat_pad(cat_masks, batch.cat_mask.dtype)
        if batch.num_mask is not None:
            batch.num_mask = cat_pad(num_masks, batch.num_mask.dtype)

    def _time_sensitive_cl(self, timestamps, lenghts):
        """
        timestamps: tensor of size (L, 1)
        lenghts: tensor of size (1)

        timestamps are padded with zeros at the end
        lenghts store number of valid timestamps for each sequence
        """
        times = torch.clone(timestamps).reshape(-1)[:lenghts]

        # Compute average interval times for each timestamp
        avg_interval_times = [
            (times[i + 1] - times[i - 1]) / 2 for i in range(1, len(times) - 1)
        ]
        avg_interval_times.insert(0, times[1] - times[0])  # First interval
        avg_interval_times.append(times[-1] - times[-2])  # Last interval

        # Create pairs of (index, time, avg_interval_time)
        pairs = [
            (idx, time, avg_interval.item())
            for idx, (time, avg_interval) in enumerate(zip(times, avg_interval_times))
        ]
        pairs.sort(key=lambda x: x[2])  # Sort by avg_interval_time
        indices = [idx for idx, time, avg_interval in pairs]

        # Determine sample length
        length = int(
            np.random.uniform(self.len_sampling_bound[0], self.len_sampling_bound[1])
            * len(times)
        )
        length = max(length, 1)

        # Split indices into dense and sparse regions
        midpoint = len(indices) // 2
        dense_indices = indices[:midpoint]
        sparse_indices = indices[midpoint:]

        random.shuffle(dense_indices)
        random.shuffle(sparse_indices)

        # Determine the number of dense and sparse samples
        dense_length = int(
            np.random.uniform(
                self.dense_sampling_bound[0], self.dense_sampling_bound[1]
            )
            * length
        )
        dense_length = max(dense_length, 1)
        sparse_length = length - dense_length

        # Select dense and sparse samples
        selected_indices = set(
            dense_indices[:dense_length] + sparse_indices[:sparse_length]
        )

        return selected_indices

    def _seg_masking(self, mask, timestamps):
        """
        - mask is a (T, D) tensor
        - timestamps is a (T) tensor
        - return: (T, D, 2) tensor
        """

        D = mask.size(1)
        interp_mask = torch.zeros_like(mask)
        sampled_times = timestamps[:, None].expand(-1, D).clone()

        sampled_times[~mask] = torch.inf
        sampled_times_start = sampled_times.amin(dim=0)
        sampled_times[~mask] = -torch.inf
        sampled_times_end = sampled_times.amax(dim=0)

        time_of_masked_segment = (
            sampled_times_end - sampled_times_start
        ) * self.mask_ratio_per_seg

        available_samples_to_sample = mask & (
            timestamps[:, None] < (sampled_times_end - time_of_masked_segment)
        )
        masking_times = False
        for _ in range(self.segment_num):
            start_time = torch.tensor(
                [
                    (
                        random.choice(timestamps[available_samples_to_sample[:, i]])
                        if available_samples_to_sample[:, i].any()
                        else torch.inf
                    )
                    for i in range(D)
                ],
                device=mask.device,
            )

            pre_time = start_time - time_of_masked_segment
            end_time = start_time + time_of_masked_segment

            chosen_times = (start_time <= timestamps[:, None]) & (
                timestamps[:, None] <= end_time
            )

            masking_times = masking_times | chosen_times

            # Update available_samples_to_sample by removing chosen_times(and times before that)
            available_samples_to_sample &= (timestamps[:, None] < pre_time) | (
                timestamps[:, None] > end_time
            )

        masking_times = masking_times & mask
        mask[masking_times] = 0.0
        interp_mask[masking_times] = 1.0

        return torch.stack([mask, interp_mask], dim=-1)

    def _time_sensitive_sampling(self, mask, timestamps):
        timestamps = timestamps.reshape(-1)  # Ensures timestamps is a 1D array

        sampled_times = mask

        if not sampled_times.any():
            return torch.tensor([])

        sampled_times_start, sampled_times_end = timestamps[sampled_times][[0, -1]]
        time_of_masked_segment = (
            sampled_times_end - sampled_times_start
        ) * self.mask_ratio_per_seg

        available_samples_to_sample = sampled_times & (
            timestamps < (sampled_times_end - time_of_masked_segment)
        )

        if not available_samples_to_sample.any():
            return torch.tensor([])

        masking_times = 0
        for _ in range(self.segment_num):
            start_time = random.choice(timestamps[available_samples_to_sample])
            pre_time = start_time - time_of_masked_segment
            end_time = start_time + time_of_masked_segment

            chosen_times = (start_time <= timestamps) & (timestamps <= end_time)

            masking_times = masking_times | chosen_times

            # Update available_samples_to_sample by removing chosen_times(and times before that)
            available_samples_to_sample &= (timestamps < pre_time) | (
                timestamps > end_time
            )

            if not available_samples_to_sample.any():
                break
        masking_times = masking_times & sampled_times
        return torch.nonzero(masking_times)

    def _constant_length_sampling(self, mask):
        count_ones = mask.sum().long().item()
        seg_seq_len = max(int(self.mask_ratio_per_seg * count_ones), 1)

        ones_indices_in_mask = torch.where(mask == 1)[0].tolist()

        seg_pos = []
        for _ in range(self.segment_num):
            if len(ones_indices_in_mask) < seg_seq_len:
                break

            start_idx = random.choice(ones_indices_in_mask[: -seg_seq_len + 1])
            start = ones_indices_in_mask.index(start_idx)
            end = start + seg_seq_len

            sub_seg = ones_indices_in_mask[start:end]
            seg_pos.extend(sub_seg)

            # Update ones_indices_in_mask by removing selected indices
            ones_indices_in_mask = sorted(set(ones_indices_in_mask) - set(sub_seg))

        return list(set(seg_pos))


class CatToNum(BatchTransform):
    """Process categorical features as numerical.

    Treat categorical features as numerical (just type cast). Category 0 is converted
    to NaN value.
    """

    def __call__(self, batch: Batch):
        if batch.cat_features_names is None or batch.cat_features is None:
            logger.warning(
                "Batch does not have categorical features, ignoring transform"
            )
            return

        new_num = torch.where(
            batch.cat_features == MISSING_CAT_VAL, torch.nan, batch.cat_features
        )
        batch.cat_features = None
        new_num_names = batch.cat_features_names
        batch.cat_features_names = None

        if batch.num_features_names is None:
            batch.num_features_names = new_num_names
            batch.num_features = new_num
            return

        assert batch.num_features is not None
        batch.num_features_names += new_num_names
        batch.num_features = torch.cat((batch.num_features, new_num), dim=2)


class MaskValid(BatchTransform):
    """Add mask indicating valid values to batch.

    Mask has shape (max_seq_len, batch_size, n_features) and has True values where there
    are non-NaN values (nonzero category) and where the data is not padded.
    """

    def __call__(self, batch: Batch):
        max_len = batch.lengths.amax().item()
        assert isinstance(max_len, int)
        len_mask = (torch.arange(max_len)[:, None] < batch.lengths)[..., None]

        if batch.num_features is not None:
            batch.num_mask = len_mask & ~torch.isnan(batch.num_features)

        if batch.cat_features is not None:
            batch.cat_mask = len_mask & (batch.cat_features != 0)
