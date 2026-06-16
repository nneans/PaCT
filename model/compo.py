"""Neural network building blocks for PaCT.

This module owns reusable PyTorch components only: patch construction,
embeddings, prefix encoders, decoder blocks, fusion, and prediction heads.
Training loops, evaluation metrics, and batch/output data contracts live in
model/model.py, utils/metrics.py, and contracts.py respectively.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization block used inside transformer layers."""

    def __init__(self, d: int, eps: float = 1e-5):
        """Create a learned RMSNorm scale for the last dimension d."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last tensor dimension and scale it with a learned weight."""
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return x_normed * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network used as the transformer MLP block."""

    def __init__(self, d_in: int, d_out: int, d_ffn: Optional[int] = None, dropout: float = 0.0):
        """Create gated feed-forward projections from d_in to d_out."""
        super().__init__()
        if d_ffn is None:
            d_ffn = d_out
        hidden_dim = int(2 * d_ffn / 3)
        self.w1 = nn.Linear(d_in, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_in, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input through gated SiLU activations and return d_out features."""
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


def _normalize_patch_size(patch_size: int) -> int:
    """Validate and normalize patch size used by prefix patch construction."""
    patch_size = max(int(patch_size), 1)
    if patch_size != 1 and patch_size % 2 != 0:
        raise ValueError("prefix_patch_size should be 1 or an even number.")
    return patch_size


def _prefix_patch_starts(seq_len: int, patch_size: int) -> list[int]:
    """Compute tail-aligned prefix patch start offsets for one sequence length.

    Input is a valid prefix length and a patch size. The output is a list of
    possibly negative starts so short prefixes are left-padded inside a patch,
    while longer prefixes are covered by half-overlapping patches ending at the
    real sequence tail.
    """
    if seq_len <= 0:
        return []
    patch_size = _normalize_patch_size(patch_size)
    stride = max(1, patch_size // 2)
    if patch_size == 1:
        return list(range(seq_len))
    if seq_len < patch_size:
        return [seq_len - patch_size]

    starts = []
    start = -stride
    while start + patch_size <= seq_len:
        starts.append(start)
        start += stride

    tail_start = seq_len - patch_size
    if not starts or starts[-1] + patch_size < seq_len:
        starts.append(tail_start)

    deduped = []
    seen = set()
    for value in starts:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _build_prefix_patch_index(
    pad_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    patch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build batched prefix patch start indices and masks.

    Inputs are a pad mask and shape metadata for a padded prefix tensor. The
    function derives valid lengths, groups identical lengths to reuse start
    offsets, and returns (patch_starts, patch_pad_mask, valid_lens) for later
    torch.gather calls.
    """
    patch_size = _normalize_patch_size(patch_size)
    if pad_mask is None:
        valid_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
    else:
        valid_lens = (~pad_mask).sum(dim=1)

    unique_lens = torch.unique(valid_lens).tolist()
    starts_by_len = {int(length): _prefix_patch_starts(int(length), patch_size) for length in unique_lens}
    max_patches = max((len(starts) for starts in starts_by_len.values()), default=1)

    sentinel = -patch_size - 1
    patch_starts = torch.full((batch_size, max_patches), sentinel, device=device, dtype=torch.long)
    for length, starts in starts_by_len.items():
        if not starts:
            continue
        row_mask = valid_lens.eq(length)
        patch_starts[row_mask, :len(starts)] = torch.tensor(starts, device=device, dtype=torch.long)

    return patch_starts, patch_starts.eq(sentinel), valid_lens


def _gather_prefix_patch_values(
    values: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    patch_size: int,
    fill_value: Union[int, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather padded token/value sequences into prefix patch tensors.

    Input values have shape [batch, seq]. The helper builds patch indices,
    gathers each patch slot, fills invalid left-padding/tail positions with the
    provided fill value, and returns [batch, n_patches, patch_size] plus the
    patch padding mask.
    """
    batch_size, seq_len = values.shape
    patch_size = _normalize_patch_size(patch_size)
    patch_starts, patch_pad_mask, valid_lens = _build_prefix_patch_index(
        pad_mask=pad_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        patch_size=patch_size,
        device=values.device,
    )

    max_patches = patch_starts.shape[1]
    if max_patches == 0:
        empty = torch.zeros(batch_size, max_patches, patch_size, device=values.device, dtype=values.dtype)
        return empty, patch_pad_mask

    slot_offsets = torch.arange(patch_size, device=values.device, dtype=torch.long).view(1, 1, patch_size)
    source_idx = patch_starts.unsqueeze(-1) + slot_offsets
    valid_source = (source_idx >= 0) & (source_idx < valid_lens.view(batch_size, 1, 1))
    safe_source_idx = source_idx.clamp(0, max(seq_len - 1, 0))

    gathered = torch.gather(values, dim=1, index=safe_source_idx.reshape(batch_size, max_patches * patch_size))
    patches = gathered.reshape(batch_size, max_patches, patch_size).masked_fill(~valid_source, fill_value)
    return patches.contiguous(), patch_pad_mask


def build_strided_prefix_patches(
    x: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert padded prefix token ids into strided, tail-aligned patch tokens.

    Input is a padded integer token matrix and optional token pad mask. Output
    is the integer patch tensor consumed by PatchEmbedding and a patch-level
    pad mask consumed by prefix channel encoders.
    """
    return _gather_prefix_patch_values(x, pad_mask, patch_size, fill_value=0)


def gather_strided_prefix_values(
    values: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    patch_size: int,
) -> torch.Tensor:
    """Convert padded continuous prefix values into strided patch values.

    This mirrors build_strided_prefix_patches for temporal channels: raw float
    values are gathered into [batch, n_patches, patch_size] with invalid slots
    filled as 0.0.
    """
    patches, _ = _gather_prefix_patch_values(values, pad_mask, patch_size, fill_value=0.0)
    return patches


def _resolve_position_ids(
    seq_len: int,
    pos_offset: Union[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Create absolute position ids from sequence positions plus scalar/batched offset."""
    base_pos = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)
    if isinstance(pos_offset, torch.Tensor):
        return base_pos + pos_offset.to(device=device, dtype=torch.long)
    return base_pos + int(pos_offset)


def _add_position_embedding(
    x: torch.Tensor,
    pos_embedding: nn.Embedding,
    pos_offset: Union[int, torch.Tensor],
    module_name: str,
) -> torch.Tensor:
    """Add learned positional embeddings after checking max_seq_len bounds."""
    pos_ids = _resolve_position_ids(x.shape[1], pos_offset, x.device)
    max_pos = int(pos_ids.max().item()) if pos_ids.numel() > 0 else -1
    if max_pos >= pos_embedding.num_embeddings:
        raise ValueError(
            f"{module_name} position index {max_pos} exceeds max_seq_len={pos_embedding.num_embeddings}. "
            f"Increase max_seq_len or shorten the effective sequence."
        )
    return x + pos_embedding(pos_ids)


class _PatchEmbedding(nn.Module):
    """Shared patch-embedding pipeline for categorical and temporal channels.

    Subclasses implement only `_embed_slots`, mapping a raw patch tensor to
    per-slot embeddings of shape [batch, n_patches, patch_size, d_emb]. This
    base normalizes those embeddings, adds slot embeddings, flattens each patch
    into one vector, projects to d_model, and adds absolute position
    embeddings, returning [batch, n_patches, d_model].
    """

    def __init__(
        self,
        d_emb: int,
        d_model: int,
        patch_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """Create the shared slot, normalization, projection, and position layers."""
        super().__init__()
        self.patch_size = int(patch_size)
        self.slot_emb = nn.Embedding(self.patch_size, d_emb)
        self.norm = RMSNorm(d_emb)
        self.proj = nn.Linear(self.patch_size * d_emb, d_model, bias=False)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def _embed_slots(self, patches: torch.Tensor) -> torch.Tensor:
        """Map a raw patch tensor to per-slot embeddings [batch, n_patches, patch_size, d_emb]."""
        raise NotImplementedError

    def forward(self, patches: torch.Tensor, pos_offset: Union[int, torch.Tensor] = 0) -> torch.Tensor:
        """Encode raw patches into d_model patch embeddings with positions."""
        bsz, seq_len = patches.shape[0], patches.shape[1]
        emb = self.norm(self.dropout(self._embed_slots(patches)))
        slot_ids = torch.arange(self.patch_size, device=patches.device)
        patch_tokens = emb + self.slot_emb(slot_ids).view(1, 1, self.patch_size, -1)
        out = self.proj(patch_tokens.reshape(bsz, seq_len, self.patch_size * emb.shape[-1]))
        out = _add_position_embedding(out, self.pos_emb, pos_offset, self.__class__.__name__)
        return self.dropout(out)


class PatchEmbedding(_PatchEmbedding):
    """Embed categorical patch tokens into model-width patch representations.

    Input patches are token ids shaped [batch, n_patches, patch_size].
    """

    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        d_model: int,
        patch_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
        shared_token_emb: Optional[nn.Embedding] = None,
    ):
        """Create the token embedding plus the shared patch pipeline."""
        super().__init__(d_emb, d_model, patch_size, max_seq_len, dropout)
        self.token_emb = shared_token_emb or nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)

    def _embed_slots(self, patches: torch.Tensor) -> torch.Tensor:
        """Look up token-id patches as per-slot embeddings."""
        return self.token_emb(patches)


class TimePatchEmbedding(_PatchEmbedding):
    """Embed continuous temporal patch values into model-width patch representations.

    Input patches are float values shaped [batch, n_patches, patch_size].
    """

    def __init__(
        self,
        d_emb: int,
        d_model: int,
        patch_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """Create the value projection plus the shared patch pipeline."""
        super().__init__(d_emb, d_model, patch_size, max_seq_len, dropout)
        self.value_proj = nn.Linear(1, d_emb, bias=False)

    def _embed_slots(self, patches: torch.Tensor) -> torch.Tensor:
        """Project float-value patches into per-slot embeddings."""
        return self.value_proj(patches.unsqueeze(-1))


class PrefixChannelEncoderBlock(nn.Module):
    """Transformer-style self-attention block for one prefix channel."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        """Create self-attention and FFN sublayers for one prefix encoder block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_model, d_ffn=d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode one channel's patch sequence while ignoring padded patches."""
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class PrefixChannelEncoderStack(nn.Module):
    """Stack multiple PrefixChannelEncoderBlock layers for one prefix channel."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        """Create repeated post-norm prefix-channel encoder blocks."""
        super().__init__()
        self.layers = nn.ModuleList(
            [PrefixChannelEncoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass patch embeddings through every prefix encoder block."""
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
        return x


class CategoricalPrefixChannel(nn.Module):
    """One categorical prefix channel: patch, embed, and encode a token sequence.

    Used for the activity channel (k=0) and each attribute channel (k>=2). The
    module owns its patch embedding and channel encoder so a raw padded token
    sequence maps directly to encoded patch states.
    """

    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        d_model: int,
        patch_size: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        shared_token_emb: Optional[nn.Embedding] = None,
    ):
        """Create the patch embedding and channel encoder for one categorical channel."""
        super().__init__()
        self.patch_size = int(patch_size)
        self.embedder = PatchEmbedding(
            vocab_size,
            d_emb,
            d_model,
            self.patch_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            shared_token_emb=shared_token_emb,
        )
        self.encoder = PrefixChannelEncoderStack(d_model, nhead, num_layers, d_ff, dropout)

    def forward(
        self,
        seq: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        patch_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode a padded token sequence into channel states."""
        patches, _ = build_strided_prefix_patches(seq, pad_mask, self.patch_size)
        return self.encoder(self.embedder(patches), pad_mask=patch_pad_mask)


class TemporalPrefixChannel(nn.Module):
    """The timestamp prefix channel (k=1): patch, project, and encode time deltas.

    Mirrors CategoricalPrefixChannel for the continuous time channel, using a
    linear value projection in place of a token embedding.
    """

    def __init__(
        self,
        d_emb: int,
        d_model: int,
        patch_size: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
    ):
        """Create the time patch embedding and channel encoder for the timestamp channel."""
        super().__init__()
        self.patch_size = int(patch_size)
        self.embedder = TimePatchEmbedding(
            d_emb=d_emb,
            d_model=d_model,
            patch_size=self.patch_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.encoder = PrefixChannelEncoderStack(d_model, nhead, num_layers, d_ff, dropout)

    def forward(
        self,
        values: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        patch_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode padded time-delta values into channel states."""
        patches = gather_strided_prefix_values(values, pad_mask, self.patch_size)
        return self.encoder(self.embedder(patches), pad_mask=patch_pad_mask)


class PrefixCrossChannelFusion(nn.Module):
    """Fuse activity, attribute, and optional temporal prefix channels."""

    def __init__(self, num_channels: int, d_model: int, d_ff: int, dropout: float = 0.1):
        """Create projection layers that fuse concatenated channel states."""
        super().__init__()
        self.norm_in = RMSNorm(num_channels * d_model)
        self.ffn = SwiGLUFFN(num_channels * d_model, d_model, d_ffn=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, channel_hiddens: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concatenate channel states, project them, and add them to activity states."""
        activity = channel_hiddens[0]
        fused = self.ffn(self.norm_in(torch.cat(channel_hiddens, dim=-1)))
        return activity + self.dropout(fused)


class TransformerDecoderBlock(nn.Module):
    """Post-norm Transformer decoder block with prefix-context cross-attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """Create masked self-attention, cross-attention, FFN, and post-norms."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SwiGLUFFN(d_model, d_model, d_ffn=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def build_causal_mask(
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a causal self-attention mask for suffix tokens."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: torch.Tensor,
        suffix_pad_mask: Optional[torch.Tensor] = None,
        context_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode suffix states with causal self-attention and prefix cross-attention."""
        self_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=self_attn_mask,
            key_padding_mask=suffix_pad_mask,
        )
        x = self.norm1(x + self.drop(self_out))
        cross_out, _ = self.cross_attn(
            x,
            context,
            context,
            key_padding_mask=context_pad_mask,
        )
        x = self.norm2(x + self.drop(cross_out))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class PatchTransformerDecoderStack(nn.Module):
    """Standard Transformer decoder stack over suffix tokens and prefix context."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """Create the cross-attention decoder block stack."""
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        prefix_context: torch.Tensor,
        suffix_tokens: torch.Tensor,
        prefix_pad_mask: Optional[torch.Tensor] = None,
        suffix_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode suffix tokens with cross-attention over the encoded prefix context."""
        suffix_len = suffix_tokens.shape[1]
        x = suffix_tokens
        attn_mask = TransformerDecoderBlock.build_causal_mask(suffix_len, x.device)
        for layer in self.layers:
            x = layer(
                x,
                prefix_context,
                self_attn_mask=attn_mask,
                suffix_pad_mask=suffix_pad_mask,
                context_pad_mask=prefix_pad_mask,
            )
        return x

class PatchedSuffixEmbedding(nn.Module):
    """Embed decoder suffix activity tokens with optional temporal values."""

    def __init__(
        self,
        vocab_size_act: int,
        d_emb: int,
        d_model: int,
        max_seq_len: int,
        mask_id: int,
        dropout: float = 0.1,
        shared_activity_emb: Optional[nn.Embedding] = None,
    ):
        """Create tied activity, mask, time, projection, and position embeddings."""
        super().__init__()
        self.vocab_size_act = int(vocab_size_act)
        self.mask_id = int(mask_id)
        self.token_emb = shared_activity_emb or nn.Embedding(vocab_size_act + 1, d_emb, padding_idx=0)
        self.mask_emb = nn.Parameter(torch.empty(d_emb))
        self.time_proj = nn.Linear(1, d_emb, bias=False)
        self.norm = RMSNorm(d_emb)
        self.proj = nn.Linear(d_emb, d_model, bias=False)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.mask_emb, mean=0.0, std=0.02)

    def forward(
        self,
        act_tokens: torch.Tensor,
        pos_offset: Union[int, torch.Tensor] = 0,
        time_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert suffix token ids into decoder embeddings.

        Input is activity ids and optional normalized time values. The method
        applies tied activity embeddings, substitutes mask embeddings when
        needed, adds temporal projections, projects to d_model, and adds
        absolute suffix positions.
        """
        is_mask = act_tokens.eq(self.mask_id)
        safe_tokens = act_tokens.masked_fill(is_mask, 0)
        if safe_tokens.numel() > 0 and int(safe_tokens.max().item()) > self.vocab_size_act:
            raise ValueError(f"Suffix token id exceeds vocab_size_act={self.vocab_size_act}")
        emb = self.token_emb(safe_tokens)
        emb = torch.where(is_mask.unsqueeze(-1), self.mask_emb.view(1, 1, -1).to(emb.dtype), emb)
        if time_tokens is not None:
            emb = emb + self.time_proj(time_tokens.unsqueeze(-1).to(emb.dtype))
        out = self.proj(self.norm(self.dropout(emb)))
        out = _add_position_embedding(out, self.pos_emb, pos_offset, self.__class__.__name__)
        return self.dropout(out)


class ActivityPredictor(nn.Module):
    """Project decoder hidden states back to activity logits with tied weights."""

    def __init__(self, d_model: int, d_emb: int, vocab_size_act: int, tied_activity_weight: nn.Parameter):
        """Create the hidden-to-embedding bridge and tied activity output head."""
        super().__init__()
        self.act_bridge = nn.Linear(d_model, d_emb, bias=False)
        self.act_head = nn.Linear(d_emb, vocab_size_act + 1, bias=False)
        self.act_head.weight = tied_activity_weight

    def forward(self, suffix_hidden: torch.Tensor) -> torch.Tensor:
        """Return activity logits for every suffix hidden state."""
        return self.act_head(self.act_bridge(suffix_hidden))


class TemporalPredictor(nn.Module):
    """Predict next-event and remaining-time regression targets from suffix states."""

    def __init__(self, d_model: int):
        """Create regression heads for next-event and remaining-time targets."""
        super().__init__()
        self.next_time_head = nn.Linear(d_model, 1)
        self.remaining_time_head = nn.Linear(d_model, 1)

    def forward(self, suffix_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-step next-time and sequence-level remaining-time predictions."""
        next_time = self.next_time_head(suffix_hidden).squeeze(-1)
        remaining_time = self.remaining_time_head(suffix_hidden[:, 0, :]).squeeze(-1)
        return next_time, remaining_time


class PaCTDecoderComponents(nn.Module):
    """Assemble PaCT encoder, decoder, and prediction heads.

    Prefix input is activity plus configured attributes and optional temporal
    values. Suffix input is activity-only teacher-forcing or rollout tokens.
    The class turns those inputs into the prefix context, suffix hidden states,
    and activity/temporal predictions.
    """

    def __init__(
        self,
        attribute_dims: Sequence[int],
        prefix_patch_size: int,
        d_emb: int,
        d_model: int,
        nhead: int,
        num_prefix_layers: int,
        num_decoder_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
    ):
        """Create all PaCT submodules from dataset vocabulary sizes and hyperparameters."""
        super().__init__()
        self.prefix_patch_size = int(prefix_patch_size)
        self.attribute_dims = list(attribute_dims)
        self.vocab_size_act = int(attribute_dims[0])
        self.mask_id = self.vocab_size_act + 1
        self.num_prefix_channels = len(attribute_dims) + 1

        # Prefix channels in paper order: k=0 activity, k=1 timestamp, k>=2
        # attributes. The activity channel owns the shared embedding that is
        # also tied into the suffix embedder and activity head; attribute
        # channels (attribute_dims[1:]) get their own categorical embeddings.
        self.shared_activity_emb = nn.Embedding(self.vocab_size_act + 1, d_emb, padding_idx=0)
        attr_dims = self.attribute_dims[1:]
        self.prefix_channels = nn.ModuleList()
        self.prefix_channels.append(
            CategoricalPrefixChannel(
                vocab_size=self.vocab_size_act,
                d_emb=d_emb,
                d_model=d_model,
                patch_size=self.prefix_patch_size,
                nhead=nhead,
                num_layers=num_prefix_layers,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
                shared_token_emb=self.shared_activity_emb,
            )
        )
        self.prefix_channels.append(
            TemporalPrefixChannel(
                d_emb=d_emb,
                d_model=d_model,
                patch_size=self.prefix_patch_size,
                nhead=nhead,
                num_layers=num_prefix_layers,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
        )
        for vocab_size in attr_dims:
            self.prefix_channels.append(
                CategoricalPrefixChannel(
                    vocab_size=vocab_size,
                    d_emb=d_emb,
                    d_model=d_model,
                    patch_size=self.prefix_patch_size,
                    nhead=nhead,
                    num_layers=num_prefix_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    shared_token_emb=None,
                )
            )

        self.prefix_fusion = PrefixCrossChannelFusion(
            num_channels=self.num_prefix_channels,
            d_model=d_model,
            d_ff=4 * d_model,
            dropout=dropout,
        )

        self.suffix_embedder = PatchedSuffixEmbedding(
            vocab_size_act=self.vocab_size_act,
            d_emb=d_emb,
            d_model=d_model,
            max_seq_len=max_seq_len,
            mask_id=self.mask_id,
            dropout=dropout,
            shared_activity_emb=self.shared_activity_emb,
        )
        self.decoder = PatchTransformerDecoderStack(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.heads = ActivityPredictor(
            d_model=d_model,
            d_emb=d_emb,
            vocab_size_act=self.vocab_size_act,
            tied_activity_weight=self.shared_activity_emb.weight,
        )
        self.temporal_heads = TemporalPredictor(d_model)

    def encode_prefix(
        self,
        act_seq: torch.Tensor,
        cat_seqs: Sequence[torch.Tensor],
        pad_mask: Optional[torch.Tensor] = None,
        prefix_time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prefix channels into the prefix context.

        Inputs are padded prefix activity ids, categorical attribute sequences,
        a token pad mask, and normalized timestamps. Each channel is patched,
        embedded, and encoded independently in paper order (k=0 activity, k=1
        timestamp, k>=2 attributes), then fused into the prefix context with a
        patch-level pad mask.
        """
        cat_seqs = list(cat_seqs or [])
        expected_attr_channels = len(self.attribute_dims) - 1
        if len(cat_seqs) != expected_attr_channels:
            raise ValueError(
                f"Expected {expected_attr_channels} attribute channels, got {len(cat_seqs)}"
            )
        if prefix_time is None:
            raise ValueError("prefix_time is required.")

        # The patch pad mask depends only on the token pad mask and patch size,
        # so it is shared across all channels; derive it once from the activity
        # sequence and reuse it everywhere.
        _, patch_pad_mask = build_strided_prefix_patches(act_seq, pad_mask, self.prefix_patch_size)

        # Channel inputs aligned with self.prefix_channels: activity, timestamp,
        # then one entry per attribute channel.
        channel_inputs = [act_seq, prefix_time, *cat_seqs]
        channel_hiddens = [
            channel(channel_input, pad_mask, patch_pad_mask)
            for channel, channel_input in zip(self.prefix_channels, channel_inputs)
        ]

        return self.prefix_fusion(channel_hiddens), patch_pad_mask

    def decode_suffix(
        self,
        prefix_context: torch.Tensor,
        suffix_act_tokens: torch.Tensor,
        prefix_pad_mask: Optional[torch.Tensor] = None,
        suffix_pad_mask: Optional[torch.Tensor] = None,
        suffix_time_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode suffix tokens against the encoded prefix context.

        The method embeds token-level suffix history, runs the Transformer
        decoder stack with prefix-context cross-attention, and returns suffix
        hidden states.
        """
        if suffix_act_tokens.dim() == 3:
            raise ValueError("Suffix patch tensors are no longer supported; pass token-level suffix ids.")
        suffix_tokens = self.suffix_embedder(suffix_act_tokens, time_tokens=suffix_time_tokens)
        hidden = self.decoder(
            prefix_context=prefix_context,
            suffix_tokens=suffix_tokens,
            prefix_pad_mask=prefix_pad_mask,
            suffix_pad_mask=suffix_pad_mask,
        )
        return hidden

    def predict_heads(self, suffix_hidden: torch.Tensor) -> torch.Tensor:
        """Convert suffix hidden states into activity logits."""
        return self.heads(suffix_hidden)

    def predict_temporal(self, suffix_hidden: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return temporal predictions when enabled, otherwise (None, None)."""
        return self.temporal_heads(suffix_hidden)