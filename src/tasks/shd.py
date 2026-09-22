"""
COMPUTATIONAL COUPLING --- Spiking Heidelberg Digits (SHD) task interface.

SHD (Cramer et al., IEEE TNNLS 2022) is a real spike-based benchmark: 20 spoken
digit classes in English and German, converted to spike trains by a biologically
grounded cochlear model, recorded from 12 speakers.  It replaces the synthetic
pattern-discrimination tasks used in earlier revisions of this project, which
saturated at 100% accuracy and therefore could not resolve differences between
architectures.

Preprocessing follows standard practice for the dataset: spikes are binned into a
fixed number of time steps, and the 700 cochlear channels are pooled into coarser
bands to keep the input dimensionality tractable.  Speaker-disjoint evaluation is
available via :func:`speaker_split`, which is the harder and more honest protocol
because the canonical test split holds out two speakers entirely.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

DEFAULT_SHD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "shd",
)

N_CHANNELS_RAW = 700
N_CLASSES = 20


@dataclass
class SpikeDataset:
    """Binned spike-count tensor ``X`` of shape (n_samples, n_steps, n_channels)."""

    X: np.ndarray          # uint8 spike counts
    y: np.ndarray          # int64 labels
    speaker: np.ndarray    # int64 speaker ids
    n_steps: int
    n_channels: int

    def __len__(self) -> int:
        return len(self.y)


def load_shd(
    split: str = "train",
    n_steps: int = 100,
    n_channels: int = 140,
    t_max: float = 1.0,
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> SpikeDataset:
    """Load and bin one SHD split.

    Parameters
    ----------
    n_steps : int
        Number of time bins spanning ``[0, t_max]`` seconds.
    n_channels : int
        Number of pooled cochlear bands; must divide 700.
    """
    import h5py

    if data_dir is None:
        data_dir = DEFAULT_SHD_DIR
    if N_CHANNELS_RAW % n_channels:
        raise ValueError(f"n_channels must divide {N_CHANNELS_RAW}, got {n_channels}")
    pool = N_CHANNELS_RAW // n_channels

    path = os.path.join(data_dir, f"shd_{split}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SHD split not found at {path}. Download from https://zenkelab.org/datasets/"
        )

    with h5py.File(path, "r") as f:
        labels = np.asarray(f["labels"][:], dtype=np.int64)
        speakers = np.asarray(f["extra"]["speaker"][:], dtype=np.int64)
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        n = len(labels) if max_samples is None else min(max_samples, len(labels))

        X = np.zeros((n, n_steps, n_channels), dtype=np.uint8)
        for i in range(n):
            t = np.asarray(times[i], dtype=np.float64)
            u = np.asarray(units[i], dtype=np.int64)
            keep = t < t_max
            t, u = t[keep], u[keep]
            if t.size == 0:
                continue
            tb = np.minimum((t / t_max * n_steps).astype(np.int64), n_steps - 1)
            ub = np.minimum(u // pool, n_channels - 1)
            flat = tb * n_channels + ub
            counts = np.bincount(flat, minlength=n_steps * n_channels)
            np.clip(counts, 0, 255, out=counts)
            X[i] = counts.reshape(n_steps, n_channels).astype(np.uint8)

    return SpikeDataset(
        X=X,
        y=labels[:n],
        speaker=speakers[:n],
        n_steps=n_steps,
        n_channels=n_channels,
    )


def speaker_split(ds: SpikeDataset, held_out_speakers: Tuple[int, ...]) -> Tuple[SpikeDataset, SpikeDataset]:
    """Split a dataset by speaker identity for speaker-disjoint validation."""
    mask = np.isin(ds.speaker, np.asarray(held_out_speakers))
    def sub(m):
        return SpikeDataset(
            X=ds.X[m], y=ds.y[m], speaker=ds.speaker[m],
            n_steps=ds.n_steps, n_channels=ds.n_channels,
        )
    return sub(~mask), sub(mask)


def cache_path(n_steps: int, n_channels: int, split: str, data_dir: Optional[str] = None) -> str:
    if data_dir is None:
        data_dir = DEFAULT_SHD_DIR
    return os.path.join(data_dir, f"cache_{split}_T{n_steps}_C{n_channels}.npz")


def load_shd_cached(split: str = "train", n_steps: int = 100, n_channels: int = 140,
                    data_dir: Optional[str] = None, **kw) -> SpikeDataset:
    """Load a split, caching the binned tensor to disk so repeated runs are cheap."""
    p = cache_path(n_steps, n_channels, split, data_dir)
    if os.path.exists(p):
        z = np.load(p)
        return SpikeDataset(X=z["X"], y=z["y"], speaker=z["speaker"],
                            n_steps=n_steps, n_channels=n_channels)
    ds = load_shd(split=split, n_steps=n_steps, n_channels=n_channels,
                  data_dir=data_dir, **kw)
    np.savez_compressed(p, X=ds.X, y=ds.y, speaker=ds.speaker)
    return ds
