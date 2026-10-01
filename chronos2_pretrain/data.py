"""
Batch construction for Chronos-2 pretraining.

A *task* is one forecasting problem: an array of shape (n_variates, length) whose rows are ordered
[targets | past-only covariates | known-future covariates], the same layout as `PreparedInput["context"]`
in `chronos.chronos2.preprocess`. A batch stacks several tasks along the batch axis and `group_ids` tells
the model which rows belong to the same task.

Where the tasks come from is configured through `data.sources`, each entry being one of:
  - type: arrow     Real series read (memory-mapped) through HF `datasets`. `path` is a directory written by
                    `Dataset.save_to_disk` or a glob of .parquet / .arrow files. `column` holds one series per
                    row, either 1-d or 2-d with shape (n_variates, length).
  - type: callable  An on-the-fly generator `fn(rng, length, **kwargs) -> Task` given as "module:function",
                    e.g., synthetic univariate generators and multivariatizers.
"""

import glob
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# Consecutive too-short draws tolerated before giving up, same as in `chronos.chronos2.dataset`.
MAX_REJECTED_SAMPLES = 10_000


@dataclass
class Task:
    series: np.ndarray  # (n_variates, length), NaN marks a missing value
    n_targets: int
    n_future_covariates: int = 0


class ArrowSource:
    """
    Real time series stored on disk.

    Arguments
    ----------
    path
        A directory written by `Dataset.save_to_disk` or a glob of .parquet / .arrow files
    column
        The column holding the series, by default "target"
    multivariate
        What to do with 2-d rows: "split" forecasts one randomly picked variate on its own, "joint" treats
        all variates as targets of a single task, by default "split"
    """

    def __init__(self, path: str, column: str = "target", multivariate: str = "split"):
        assert multivariate in {"split", "joint"}, f"Invalid multivariate mode: {multivariate}"
        self.path = path
        self.column = column
        self.multivariate = multivariate
        self._dataset = None

    @property
    def dataset(self):
        # opened on first use, so that every dataloader worker creates its own handle
        if self._dataset is None:
            import datasets

            if (Path(self.path) / "state.json").exists():
                dataset = datasets.load_from_disk(self.path)
            else:
                files = sorted(glob.glob(self.path, recursive=True))
                if len(files) == 0:
                    raise FileNotFoundError(f"No files match {self.path}")
                builder = "parquet" if files[0].endswith(".parquet") else "arrow"
                dataset = datasets.load_dataset(builder, data_files=files, split="train")
            self._dataset = dataset.select_columns([self.column]).with_format("numpy")
        return self._dataset

    def sample(self, rng: np.random.Generator, length: int) -> Task:
        dataset = self.dataset
        series = np.asarray(dataset[int(rng.integers(len(dataset)))][self.column], dtype=np.float32)
        if series.ndim == 1:
            series = series[None]
        elif self.multivariate == "split":
            series = series[int(rng.integers(len(series)))][None]
        return Task(series=series, n_targets=len(series))


class CallableSource:
    """
    Tasks generated on the fly.

    Arguments
    ----------
    fn
        A "module:function" reference to `fn(rng, length, **kwargs) -> Task`, where `length` is the number
        of time steps needed for a full window (context + horizon)
    **kwargs
        Forwarded to `fn`
    """

    def __init__(self, fn: str, **kwargs):
        module, _, name = fn.partition(":")
        self.fn = getattr(importlib.import_module(module), name)
        self.kwargs = kwargs

    def sample(self, rng: np.random.Generator, length: int) -> Task:
        return self.fn(rng, length, **self.kwargs)


SOURCE_TYPES = {"arrow": ArrowSource, "callable": CallableSource}


def toy_task(rng: np.random.Generator, length: int, max_variates: int = 4) -> Task:
    """Noisy sinusoids whose targets are linear mixtures of all variates. Only meant for smoke tests."""
    n_variates = int(rng.integers(1, max_variates + 1))
    n_targets = int(rng.integers(1, n_variates + 1))
    n_future_covariates = int(rng.integers(0, n_variates - n_targets + 1))

    time = np.arange(length, dtype=np.float32)
    period = rng.choice([7.0, 12.0, 24.0, 96.0, 168.0], size=(n_variates, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(n_variates, 1))
    series = np.sin(2 * np.pi * time / period + phase) + 0.1 * rng.standard_normal((n_variates, length))
    series[:n_targets] = rng.standard_normal((n_targets, n_variates)) @ series

    return Task(series=series.astype(np.float32), n_targets=n_targets, n_future_covariates=n_future_covariates)


def subsample_variates(task: Task, max_variates: int, rng: np.random.Generator) -> Task:
    """Randomly drops variates of a task which has more than `max_variates`, keeping at least one target."""
    n_variates = len(task.series)
    if n_variates <= max_variates:
        return task

    keep = np.sort(rng.choice(n_variates, size=max_variates, replace=False))
    if keep[0] >= task.n_targets:
        # no target was picked, swap the first covariate for a target (keeps the indices sorted)
        keep[0] = rng.integers(task.n_targets)

    return Task(
        series=task.series[keep],
        n_targets=int((keep < task.n_targets).sum()),
        n_future_covariates=int((keep >= n_variates - task.n_future_covariates).sum()),
    )


class PretrainingDataset(IterableDataset):
    """
    An infinite stream of training batches in the format expected by `Chronos2Model.forward`.

    Arguments
    ----------
    sources
        Source specifications, see the module docstring. The optional `weight` of a source sets how often
        it is drawn relative to the others, by default 1
    context_length
        The maximum context length of the sampled windows
    batch_size
        The number of time series in every batch. Note that the batch size here means the number of time
        series, including target(s) and covariates, that are input into the model.
    patch_size
        The output patch size of the model
    max_output_patches
        The number of output patches, i.e., the prediction horizon, is randomly sampled for each batch
        from 1 to this value
    min_past
        The minimum number of time steps the context must have. Time series shorter than
        `min_past + patch_size` are rejected.
    max_group_size
        The maximum number of time series in a task, larger tasks are randomly subsampled
    seed
        Entropy of the sampler, e.g., [seed, rank, step]. The dataloader worker id is appended to it.
    """

    def __init__(
        self,
        sources: Sequence[dict],
        context_length: int,
        batch_size: int,
        patch_size: int,
        max_output_patches: int,
        min_past: int,
        max_group_size: int,
        seed: Sequence[int],
    ) -> None:
        super().__init__()
        assert len(sources) > 0, "At least one data source is required"
        assert max_output_patches >= 1 and max_group_size >= 1

        self.sources = []
        for spec in sources:
            spec = {k: v for k, v in spec.items() if k != "weight"}
            self.sources.append(SOURCE_TYPES[spec.pop("type")](**spec))
        weights = np.array([spec.get("weight", 1.0) for spec in sources], dtype=np.float64)
        self.weights = weights / weights.sum()

        self.context_length = context_length
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_output_patches = max_output_patches
        self.min_past = min_past
        self.max_group_size = max_group_size
        self.seed = list(seed)

    def _construct_slice(
        self, task: Task, horizon: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_variates, length = task.series.shape

        # The forecast start is drawn such that the full horizon is observed whenever the series is long enough.
        # For shorter series at least one patch is observed, the rest of the horizon is padded with NaNs,
        # which are ignored during loss computation.
        if length >= self.min_past + horizon:
            last_slice_idx = length - horizon
        else:
            last_slice_idx = length - self.patch_size
        slice_idx = int(rng.integers(self.min_past, last_slice_idx + 1))
        start_idx = max(0, slice_idx - self.context_length)
        available = min(horizon, length - slice_idx)

        # non-finite values are treated as missing
        window = task.series[:, start_idx : slice_idx + available]
        window = np.where(np.isfinite(window), window, np.nan).astype(np.float32, copy=False)

        context = window[:, : slice_idx - start_idx]
        future = np.full((n_variates, horizon), np.nan, dtype=np.float32)
        future[:, :available] = window[:, slice_idx - start_idx :]

        # mask out all rows corresponding to covariates
        future_target = future.copy()
        future_target[task.n_targets :] = np.nan

        # only the trailing n_future_covariates rows are known into the future
        future_covariates = np.full_like(future, np.nan)
        if task.n_future_covariates > 0:
            future_covariates[-task.n_future_covariates :] = future[-task.n_future_covariates :]

        return context, future_target, future_covariates

    def _build_batch(self, num_output_patches: int, rng: np.random.Generator) -> dict[str, torch.Tensor | int]:
        horizon = num_output_patches * self.patch_size
        contexts, future_targets, future_covariates, group_ids = [], [], [], []

        n_rows = 0
        n_rejected = 0
        while n_rows < self.batch_size:
            source = self.sources[rng.choice(len(self.sources), p=self.weights)]
            task = source.sample(rng, self.context_length + horizon)
            if task.series.shape[-1] < self.min_past + self.patch_size:
                n_rejected += 1
                if n_rejected >= MAX_REJECTED_SAMPLES:
                    raise ValueError(
                        f"Could not sample a time series with at least min_past + patch_size "
                        f"({self.min_past + self.patch_size}) observations after {MAX_REJECTED_SAMPLES} attempts."
                    )
                continue
            n_rejected = 0

            # the task which fills up the batch is cut down to the remaining rows, so that the batch size is exact
            task = subsample_variates(task, min(self.max_group_size, self.batch_size - n_rows), rng)
            context, future_target, future_covariate = self._construct_slice(task, horizon, rng)

            group_ids.append(np.full(len(context), fill_value=len(contexts), dtype=np.int64))
            contexts.append(context)
            future_targets.append(future_target)
            future_covariates.append(future_covariate)
            n_rows += len(context)

        # left pad the contexts to the length of the longest one
        max_len = max(context.shape[-1] for context in contexts)
        batch_context = np.full((n_rows, max_len), np.nan, dtype=np.float32)
        row = 0
        for context in contexts:
            batch_context[row : row + len(context), max_len - context.shape[-1] :] = context
            row += len(context)

        return {
            "context": torch.from_numpy(batch_context),
            "future_target": torch.from_numpy(np.concatenate(future_targets)),
            "future_covariates": torch.from_numpy(np.concatenate(future_covariates)),
            "group_ids": torch.from_numpy(np.concatenate(group_ids)),
            "num_output_patches": num_output_patches,
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        """
        Yields
        ------
        dict
            A dictionary containing:
            - context: torch.Tensor of shape (batch_size, <= context_length), left padded with NaNs
            - future_target: torch.Tensor of shape (batch_size, horizon), NaN for covariates
            - future_covariates: torch.Tensor of shape (batch_size, horizon), NaN for targets and past-only covariates
            - group_ids: torch.Tensor of shape (batch_size,) containing the task of each time series
            - num_output_patches: int indicating the number of patches the model should output to cover the horizon
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng([*self.seed, worker_id])

        while True:
            num_output_patches = int(rng.integers(1, self.max_output_patches + 1))
            yield self._build_batch(num_output_patches, rng)
