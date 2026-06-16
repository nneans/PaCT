"""Shared data contracts for PaCT model inputs and outputs.

Dataclasses in this module define tensor fields exchanged between datasets,
the model, and loss/evaluation code. They intentionally contain no model
layers, training logic, or metric implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class PaCTBatch:
    """Mini-batch tensor contract consumed by PaCTNet and compute_pact_loss.

    Dataset collation converts variable-length prefix/suffix samples into these
    padded tensors. Prefix fields feed the encoder, suffix fields feed teacher
    forcing and loss masks, and temporal fields are optional regression targets.
    """
    prefix_act: torch.Tensor
    prefix_cat: Sequence[torch.Tensor]
    prefix_pad_mask: torch.Tensor
    suffix_act: torch.Tensor
    suffix_pad_mask: torch.Tensor
    prefix_time: Optional[torch.Tensor] = None
    suffix_next_time: Optional[torch.Tensor] = None
    suffix_remaining_time: Optional[torch.Tensor] = None


@dataclass
class PaCTOutput:
    """Model output contract returned by PaCTNet.forward_train.

    The activity logits, loss mask, and temporal predictions are produced by
    PaCT training/evaluation forwards.
    """
    act_logits: torch.Tensor
    loss_mask: torch.Tensor
    next_time_pred: Optional[torch.Tensor] = None
    remaining_time_pred: Optional[torch.Tensor] = None
