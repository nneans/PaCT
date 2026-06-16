from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from contracts import PaCTBatch


class TimeNormalizer:
    """Normalize raw time deltas into z-scores and invert them back to seconds.

    fit() receives training-set second values and stores mean/std. transform()
    maps one raw value into normalized model input/target space, while inverse()
    converts predictions back to non-negative seconds for reporting.
    """
    def __init__(self):
        """Initialize identity-like normalization until fit() estimates statistics."""
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values: Sequence[float]) -> None:
        """Estimate mean/std from raw second values used by temporal PaCT."""
        vals = list(values)
        if not vals:
            vals = [0.0]
        raw_vals = np.array(vals, dtype=np.float64)
        self.mean = float(np.mean(raw_vals))
        self.std = float(np.std(raw_vals)) + 1e-8

    def transform(self, value: float) -> float:
        """Convert one raw second value into a normalized float target."""
        return float((max(0.0, float(value)) - self.mean) / self.std)

    def inverse(self, z):
        """Convert normalized predictions back into non-negative seconds."""
        seconds = np.asarray(z) * self.std + self.mean
        return np.maximum(0.0, seconds)


def fit_label_encoder(values: Sequence[str]) -> LabelEncoder:
    """Fit a sklearn LabelEncoder from string labels used in activity/attribute channels."""
    enc = LabelEncoder()
    enc.fit(list(values))
    return enc


def safe_transform_with_unk(enc: LabelEncoder, vals: Sequence[str]) -> List[int]:
    """Transform strings to label ids and map unseen values to an extra UNK id."""
    mapping = {val: idx for idx, val in enumerate(enc.classes_)}
    unk_idx = len(enc.classes_)
    return [mapping.get(v, unk_idx) for v in vals]


def pad_sequences_1d(
    seqs: Sequence[torch.Tensor],
    max_len: int,
    dtype: torch.dtype,
    pad_val: float = 0,
) -> torch.Tensor:
    """Pad a list of 1D tensors into a [batch, max_len] tensor filled with pad_val.

    Shared by dataset collation and benchmark rollout so prefix/suffix padding
    is built identically in training and inference.
    """
    out = torch.full((len(seqs), max_len), pad_val, dtype=dtype)
    for i, seq in enumerate(seqs):
        out[i, : seq.shape[0]] = seq
    return out


def pad_mask_from_ids(padded_ids: torch.Tensor) -> torch.Tensor:
    """Return the padding mask (True where padded) for a padded id tensor.

    PaCT reserves id 0 for padding, so padded positions are exactly the zeros.
    """
    return padded_ids.eq(0)


class PaCTDataset(Dataset):
    """Activity-suffix dataset for workshop version.

    Prefix uses:
    - activity
    - categorical attributes

    Suffix rollout uses:
    - activity only
    """

    def __init__(
        self,
        samples: List[Dict[str, object]],
        attribute_dims: Sequence[int],
        eos_id: int,
        sos_id: int,
        encoders: Dict[str, object],
        cat_keys: Sequence[str],
    ):
        """Store prebuilt prefix/suffix samples and metadata required by PaCT.

        samples already contain tensorized prefix activity, categorical prefix
        tensors, suffix activity targets, and optional temporal tensors. The
        metadata is reused by the model to size embeddings and decode EOS/SOS.
        """
        self.samples = samples
        self.attribute_dims = list(attribute_dims)
        self.eos_id = int(eos_id)
        self.sos_id = int(sos_id)
        self.encoders = encoders
        self.cat_keys = list(cat_keys)

    def __len__(self) -> int:
        """Return the number of prefix/suffix training or evaluation samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return one raw variable-length sample before padding/collation."""
        s = self.samples[idx]
        return (
            s["prefix_act"],
            s["prefix_cat"],
            s["suffix_act"],
            s.get("prefix_time"),
            s.get("suffix_next_time"),
            s.get("suffix_remaining_time"),
        )

    def collate_fn(self, batch):
        """Pad variable-length samples into a PaCTBatch.

        Input is a list of tuples returned by __getitem__. The method pads
        activity, categorical, and optional temporal sequences to batch maxima,
        builds pad masks from zero ids, and returns the PaCTBatch consumed by
        PaCTNet.forward_train and rollout/evaluation code.
        """
        (
            prefix_acts,
            prefix_cats_list,
            suffix_acts,
            prefix_times,
            suffix_next_times,
            suffix_remaining_times,
        ) = zip(*batch)

        max_lp = max(x.shape[0] for x in prefix_acts)
        max_ls = max(x.shape[0] for x in suffix_acts)

        prefix_act = pad_sequences_1d(prefix_acts, max_lp, torch.long, pad_val=0)
        prefix_pad_mask = pad_mask_from_ids(prefix_act)

        n_cat = len(prefix_cats_list[0]) if prefix_cats_list else 0
        prefix_cat = [
            pad_sequences_1d([row[k] for row in prefix_cats_list], max_lp, torch.long, pad_val=0)
            for k in range(n_cat)
        ]

        suffix_act = pad_sequences_1d(suffix_acts, max_ls, torch.long, pad_val=0)
        suffix_pad_mask = pad_mask_from_ids(suffix_act)

        def maybe_pad_float(seqs: Sequence[Optional[torch.Tensor]], max_len: int) -> Optional[torch.Tensor]:
            """Pad optional temporal sequences, returning None when absent."""
            if not seqs or seqs[0] is None:
                return None
            return pad_sequences_1d(seqs, max_len, torch.float, pad_val=0.0)

        def maybe_stack_float(seqs: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            """Stack optional per-sample scalar/short temporal targets when present."""
            if not seqs or seqs[0] is None:
                return None
            return torch.stack([s.to(torch.float) for s in seqs], dim=0)

        return PaCTBatch(
            prefix_act=prefix_act,
            prefix_cat=prefix_cat,
            prefix_pad_mask=prefix_pad_mask,
            suffix_act=suffix_act,
            suffix_pad_mask=suffix_pad_mask,
            prefix_time=maybe_pad_float(prefix_times, max_lp),
            suffix_next_time=maybe_pad_float(suffix_next_times, max_ls),
            suffix_remaining_time=maybe_stack_float(suffix_remaining_times),
        )