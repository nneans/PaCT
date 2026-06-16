"""High-level PaCT model, training loop, rollout, and dataset evaluation.

This module orchestrates components from model/compo.py and uses the shared
batch/output contracts from contracts.py. Low-level network blocks and pure
sequence metrics stay out of this file unless they are part of model training
or evaluation control flow.
"""
from __future__ import annotations

import copy
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.compo import PaCTDecoderComponents
from contracts import PaCTBatch, PaCTOutput


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible PaCT runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest" if deterministic else "high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = not deterministic
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = not deterministic


def init_transformer_weights(module: nn.Module) -> None:
    """Initialize Linear and Embedding modules with transformer-style weights."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()


class PaCTNet(nn.Module):
    """Torch module that connects PaCT components for training and rollout.

    Inputs are PaCTBatch tensors from pact_dataset. The module encodes prefix
    activity/attribute/time channels, decodes suffix tokens, and returns either
    teacher-forced logits for loss computation or autoregressive generations.
    """

    def __init__(
        self,
        attribute_dims: Sequence[int],
        sos_id: int,
        prefix_patch_size: int,
        d_emb: int,
        d_model: int,
        nhead: int,
        num_prefix_layers: int,
        num_decoder_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        components_cls=PaCTDecoderComponents,
    ):
        """Create PaCT decoder components for a concrete dataset vocabulary."""
        super().__init__()
        self.components = components_cls(
            attribute_dims=attribute_dims,
            prefix_patch_size=prefix_patch_size,
            d_emb=d_emb,
            d_model=d_model,
            nhead=nhead,
            num_prefix_layers=num_prefix_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.sos_id = int(sos_id)

    def encode_prefix(
        self,
        prefix_act: torch.Tensor,
        prefix_cat: Sequence[torch.Tensor],
        prefix_pad_mask: torch.Tensor,
        prefix_time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode padded prefix tensors into the prefix context and its pad mask."""
        return self.components.encode_prefix(
            act_seq=prefix_act,
            cat_seqs=prefix_cat,
            pad_mask=prefix_pad_mask,
            prefix_time=prefix_time,
        )

    def _decode_suffix_hidden(
        self,
        prefix_context: torch.Tensor,
        prefix_context_pad_mask: torch.Tensor,
        suffix_input: torch.Tensor,
        suffix_pad_mask: Optional[torch.Tensor],
        suffix_input_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode token-level suffix inputs into hidden states."""
        return self.components.decode_suffix(
            prefix_context=prefix_context,
            suffix_act_tokens=suffix_input,
            prefix_pad_mask=prefix_context_pad_mask,
            suffix_pad_mask=suffix_pad_mask,
            suffix_time_tokens=suffix_input_time,
        )

    def forward_train(self, batch: PaCTBatch) -> PaCTOutput:
        """Run one teacher-forced training/evaluation forward pass.

        Input is a padded PaCTBatch. The method shifts suffix targets right with
        SOS for decoder input, shifts temporal next-time targets, and returns
        activity logits plus temporal predictions for loss.
        """
        prefix_context, prefix_context_pad_mask = self.encode_prefix(
            prefix_act=batch.prefix_act,
            prefix_cat=batch.prefix_cat,
            prefix_pad_mask=batch.prefix_pad_mask,
            prefix_time=batch.prefix_time,
        )
        suffix_input = torch.zeros_like(batch.suffix_act)
        suffix_input[:, 0] = self.sos_id
        if batch.suffix_act.shape[1] > 1:
            suffix_input[:, 1:] = batch.suffix_act[:, :-1]
        suffix_input = suffix_input.masked_fill(batch.suffix_pad_mask, 0)
        if batch.prefix_time is None or batch.suffix_next_time is None:
            raise ValueError("PaCT requires prefix_time and suffix_next_time.")
        suffix_input_time = torch.zeros_like(batch.suffix_next_time)
        if batch.suffix_next_time.shape[1] > 1:
            suffix_input_time[:, 1:] = batch.suffix_next_time[:, :-1]
        suffix_input_time = suffix_input_time.masked_fill(batch.suffix_pad_mask, 0.0)
        loss_mask = ~batch.suffix_pad_mask
        suffix_hidden = self._decode_suffix_hidden(
            prefix_context=prefix_context,
            prefix_context_pad_mask=prefix_context_pad_mask,
            suffix_input=suffix_input,
            suffix_pad_mask=batch.suffix_pad_mask,
            suffix_input_time=suffix_input_time,
        )
        act_logits = self.components.predict_heads(suffix_hidden)
        next_time_pred, remaining_time_pred = self.components.predict_temporal(suffix_hidden)
        return PaCTOutput(
            act_logits=act_logits,
            loss_mask=loss_mask,
            next_time_pred=next_time_pred,
            remaining_time_pred=remaining_time_pred,
        )

    @torch.no_grad()
    def rollout(
        self,
        prefix_act: torch.Tensor,
        prefix_cat: Sequence[torch.Tensor],
        prefix_pad_mask: torch.Tensor,
        eos_id: int,
        sos_id: int,
        max_steps: int,
        prefix_time: Optional[torch.Tensor] = None,
        return_temporal: bool = False,
    ):
        """Generate suffix activity tokens autoregressively from a prefix.

        Input is a padded prefix batch and EOS/SOS ids. The method repeatedly
        decodes the current generated suffix, masks invalid PAD/SOS logits,
        appends the argmax activity, stops on EOS, and optionally returns
        temporal rollout predictions beside generated activities.
        """
        device = prefix_act.device
        bsz = prefix_act.shape[0]
        prefix_context, prefix_context_pad_mask = self.encode_prefix(
            prefix_act=prefix_act,
            prefix_cat=prefix_cat,
            prefix_pad_mask=prefix_pad_mask,
            prefix_time=prefix_time,
        )
        suffix_input = torch.full((bsz, 1), int(sos_id), dtype=torch.long, device=device)
        suffix_time_input = torch.zeros(bsz, 1, dtype=torch.float, device=device)
        generated = []
        next_time_preds = []
        remaining_time_pred = None
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(int(max_steps)):
            suffix_hidden = self.components.decode_suffix(
                prefix_context=prefix_context,
                suffix_act_tokens=suffix_input,
                prefix_pad_mask=prefix_context_pad_mask,
                suffix_pad_mask=None,
                suffix_time_tokens=suffix_time_input,
            )
            act_logits = self.components.predict_heads(suffix_hidden)
            next_time_pred, current_remaining_pred = self.components.predict_temporal(suffix_hidden)
            if return_temporal and next_time_pred is not None:
                next_time_preds.append(next_time_pred[:, -1])
                if remaining_time_pred is None:
                    remaining_time_pred = current_remaining_pred
            step_logits = act_logits[:, -1, :].clone()
            # The activity head includes EOS and SOS because vocab_size_act is
            # set from sos_id. EOS is a valid output; PAD and SOS are not.
            step_logits[..., 0] = torch.finfo(step_logits.dtype).min
            if 0 <= int(sos_id) < step_logits.shape[-1]:
                step_logits[..., int(sos_id)] = torch.finfo(step_logits.dtype).min
            step_act = torch.argmax(step_logits, dim=-1)
            step_act = step_act.masked_fill(finished, 0)
            generated.append(step_act)
            finished = finished | step_act.eq(eos_id)
            step_time = next_time_pred[:, -1].detach().masked_fill(finished, 0.0)
            suffix_time_input = torch.cat([suffix_time_input, step_time.unsqueeze(1)], dim=1)
            suffix_input = torch.cat([suffix_input, step_act.unsqueeze(1)], dim=1)
            if finished.all():
                break

        if not generated:
            gen_act = torch.zeros(bsz, 0, dtype=torch.long, device=device)
        else:
            gen_act = torch.stack(generated, dim=1)
        if not return_temporal:
            return gen_act
        if next_time_preds:
            next_time = torch.stack(next_time_preds, dim=1)
        else:
            next_time = None
        return gen_act, next_time, remaining_time_pred

def compute_pact_loss(
    output: PaCTOutput,
    batch: PaCTBatch,
    eos_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PaCT training loss and scalar logging metrics.

    Input is PaCTNet.forward_train output plus the original batch. The activity
    loss is cross entropy over valid suffix targets. When temporal heads and
    targets exist, Huber losses for next-time and remaining-time are added.
    """
    loss_flat = F.cross_entropy(
        output.act_logits.reshape(-1, output.act_logits.shape[-1]),
        batch.suffix_act.reshape(-1),
        ignore_index=0,
        reduction="none",
    )
    loss_mask = output.loss_mask.reshape(-1) & batch.suffix_act.reshape(-1).ne(0)
    if not bool(loss_mask.any().item()):
        raise ValueError("PaCT loss_mask selected no valid suffix targets.")
    loss_act = loss_flat[loss_mask].mean()
    loss_total = loss_act
    metrics = {
        "loss_total": float(loss_act.item()),
        "loss_act": float(loss_act.item()),
        "loss_tokens": float(loss_mask.sum().item()),
    }
    if (
        output.next_time_pred is not None
        and output.remaining_time_pred is not None
        and batch.suffix_next_time is not None
        and batch.suffix_remaining_time is not None
        and eos_id is not None
    ):
        valid = output.loss_mask & batch.suffix_act.ne(0)
        next_valid = valid & batch.suffix_act.ne(int(eos_id))
        if bool(next_valid.any().item()):
            loss_next = F.huber_loss(
                output.next_time_pred[next_valid],
                batch.suffix_next_time[next_valid],
                delta=1.0,
            )
        else:
            loss_next = output.act_logits.new_zeros(())
        loss_remaining = F.huber_loss(
            output.remaining_time_pred,
            batch.suffix_remaining_time,
            delta=1.0,
        )
        loss_total = loss_total + loss_next + loss_remaining
        metrics.update({
            "loss_total": float(loss_total.item()),
            "loss_next_time": float(loss_next.item()),
            "loss_remaining_time": float(loss_remaining.item()),
            "loss_next_tokens": float(next_valid.sum().item()),
            "loss_remaining_tokens": float(batch.suffix_remaining_time.shape[0]),
        })
    return loss_total, metrics


class PaCT:
    """High-level estimator wrapper around PaCTNet.

    This class owns hyperparameters, device/data-loader setup, model building,
    fitting with early stopping, and dataset evaluation. experiment.py uses
    it as the train/evaluate unit for each fold.
    """

    name = "PaCT"

    def __init__(
        self,
        prefix_patch_size: Optional[int] = None,
        d_emb: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_prefix_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        n_epochs: int = 30,
        batch_size: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 8,
        seed: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = True,
        use_amp: bool = True,
        deterministic: bool = False,
    ):
        """Store training/model hyperparameters and initialize empty runtime state."""
        if prefix_patch_size is None:
            prefix_patch_size = 1
        if int(prefix_patch_size) != 1 and int(prefix_patch_size) % 2 != 0:
            raise ValueError("prefix_patch_size should be 1 or an even number.")
        self.prefix_patch_size = int(prefix_patch_size)
        self.d_emb = d_emb
        self.d_model = d_model
        self.nhead = nhead
        self.num_prefix_layers = num_prefix_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        default_num_workers = 0 if os.name == "nt" else max(0, min(4, (torch.get_num_threads() or 1)))
        self.num_workers = default_num_workers if num_workers is None else int(num_workers)
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.use_amp = bool(use_amp)
        self.deterministic = bool(deterministic)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[PaCTNet] = None
        self.attribute_dims: Optional[Sequence[int]] = None
        self.eos_id: Optional[int] = None
        self.sos_id: Optional[int] = None
        self.encoders: Optional[Dict[str, object]] = None

        if seed is not None:
            set_seed(seed, deterministic=deterministic)

    def _to_device(self, batch: PaCTBatch) -> PaCTBatch:
        """Move every tensor in a PaCTBatch to the configured device."""
        non_blocking = self.pin_memory and self.device.type == "cuda"
        return PaCTBatch(
            prefix_act=batch.prefix_act.to(self.device, non_blocking=non_blocking),
            prefix_cat=[x.to(self.device, non_blocking=non_blocking) for x in batch.prefix_cat],
            prefix_pad_mask=batch.prefix_pad_mask.to(self.device, non_blocking=non_blocking),
            suffix_act=batch.suffix_act.to(self.device, non_blocking=non_blocking),
            suffix_pad_mask=batch.suffix_pad_mask.to(self.device, non_blocking=non_blocking),
            prefix_time=None if batch.prefix_time is None else batch.prefix_time.to(self.device, non_blocking=non_blocking),
            suffix_next_time=None if batch.suffix_next_time is None else batch.suffix_next_time.to(self.device, non_blocking=non_blocking),
            suffix_remaining_time=None if batch.suffix_remaining_time is None else batch.suffix_remaining_time.to(self.device, non_blocking=non_blocking),
        )

    def _build(self, train_dataset) -> None:
        """Instantiate PaCTNet from dataset metadata and initialize weights."""
        self.attribute_dims = list(train_dataset.attribute_dims)
        self.eos_id = int(train_dataset.eos_id)
        self.sos_id = int(train_dataset.sos_id)
        self.encoders = train_dataset.encoders
        self.net = PaCTNet(
            attribute_dims=self.attribute_dims,
            sos_id=self.sos_id,
            prefix_patch_size=self.prefix_patch_size,
            d_emb=self.d_emb,
            d_model=self.d_model,
            nhead=self.nhead,
            num_prefix_layers=self.num_prefix_layers,
            num_decoder_layers=self.num_decoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
        ).to(self.device)
        self.net.apply(init_transformer_weights)

    def _loader_kwargs(self) -> Dict[str, object]:
        """Return DataLoader worker/pinning options compatible with the device."""
        kwargs: Dict[str, object] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory and self.device.type == "cuda",
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
        return kwargs

    def _autocast_enabled(self) -> bool:
        """Return whether CUDA automatic mixed precision should be enabled."""
        return self.use_amp and self.device.type == "cuda"

    def _build_optimizer(self) -> optim.Optimizer:
        """Create AdamW, using fused CUDA implementation when available."""
        assert self.net is not None
        adamw_kwargs = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
        if self.device.type == "cuda":
            try:
                return optim.AdamW(self.net.parameters(), fused=True, **adamw_kwargs)
            except TypeError:
                pass
        return optim.AdamW(self.net.parameters(), **adamw_kwargs)

    @torch.no_grad()
    def _evaluate_loss(self, dataset_like) -> Dict[str, float]:
        """Evaluate average validation loss over a Dataset or Subset."""
        assert self.net is not None
        collate_fn = dataset_like.dataset.collate_fn if hasattr(dataset_like, "dataset") else dataset_like.collate_fn
        loader = DataLoader(
            dataset_like,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )

        running = {"loss_total": 0.0, "loss_act": 0.0, "loss_tokens": 0.0}
        steps = 0

        self.net.eval()
        for batch in loader:
            batch = self._to_device(batch)
            with amp.autocast(device_type=self.device.type, enabled=self._autocast_enabled()):
                output = self.net.forward_train(batch)
                _, metrics = compute_pact_loss(output, batch, self.eos_id)
            steps += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

        if steps == 0:
            return running
        return {key: value / steps for key, value in running.items()}

    def fit(
        self,
        train_dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> "PaCT":
        """Train PaCT on a PaCTDataset with optional validation early stopping.

        Input is a train dataset and an optional validation dataset. The method
        builds loaders, trains with AMP and gradient clipping, chooses the best
        checkpoint by validation loss, restores that state, and stores
        timing/epoch metadata used by benchmark result rows.
        """
        self._build(train_dataset)
        assert self.net is not None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            **self._loader_kwargs(),
        )
        optimizer = self._build_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.n_epochs,
            eta_min=self.lr / 10.0,
        )
        scaler = amp.GradScaler("cuda", enabled=self._autocast_enabled())

        best_state = copy.deepcopy(self.net.state_dict())
        best_score = float("-inf")
        best_epoch = 0
        no_improve = 0
        epoch_times = []
        val_times = []
        n_epochs_run = 0

        for epoch in range(1, self.n_epochs + 1):
            epoch_start = time.perf_counter()
            n_epochs_run = epoch
            self.net.train()
            running = {"loss_total": 0.0, "loss_act": 0.0, "loss_tokens": 0.0}
            steps = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{self.n_epochs}", leave=False)
            for batch in progress:
                batch = self._to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type=self.device.type, enabled=self._autocast_enabled()):
                    output = self.net.forward_train(batch)
                    loss, metrics = compute_pact_loss(output, batch, self.eos_id)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                steps += 1
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value
                progress.set_postfix(loss=f"{running['loss_total'] / steps:.4f}")

            train_loss = running["loss_total"] / max(steps, 1)
            train_act = running["loss_act"] / max(steps, 1)

            if val_dataset is None:
                tqdm.write(f"Epoch {epoch}/{self.n_epochs} | loss_total={train_loss:.4f} loss_act={train_act:.4f}")
                epoch_times.append(time.perf_counter() - epoch_start)
                scheduler.step()
                continue

            val_start = time.perf_counter()
            loss_metrics = self._evaluate_loss(val_dataset)
            val_times.append(time.perf_counter() - val_start)
            val_loss = float(loss_metrics.get("loss_total", 0.0))
            score = -val_loss
            improved = score > best_score

            if improved:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(self.net.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            metric_msg = (
                f"val_loss={-score:.4f} | "
                f"best_val_loss={-best_score:.4f} (epoch {best_epoch})"
            )

            tqdm.write(
                f"Epoch {epoch}/{self.n_epochs} | "
                f"loss_total={train_loss:.4f} loss_act={train_act:.4f} | "
                f"{metric_msg}"
            )
            epoch_times.append(time.perf_counter() - epoch_start)

            if no_improve >= self.early_stopping_patience:
                tqdm.write(f"[Early stopping] Best val_loss={-best_score:.4f} at epoch {best_epoch}")
                break

            scheduler.step()

        self.net.load_state_dict(best_state)
        self.best_epoch_ = int(best_epoch)
        self.n_epochs_run_ = int(n_epochs_run)
        self.train_time_per_epoch_ = float(np.mean(epoch_times)) if epoch_times else 0.0
        self.val_time_sec_ = float(np.sum(val_times)) if val_times else 0.0
        return self