"""Experiment orchestration, CSV I/O, checkpointing, and evaluation for PaCT.

This module owns everything between data loading (data_io.py) and the raw
model training/rollout (model.py): KFold splits, PaCTWrapper, Evaluator,
result CSV formatting, and checkpoint save/load. The thin main.py entry point
imports run_experiment from here.
"""
from __future__ import annotations

import csv
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold

from utils.config import DATASET_CONFIG, DATASET_ORDER, FILENAME_TO_CONFIG_KEY
from utils.data_io import (
    EOC_TOKEN,
    SECONDS_PER_DAY,
    _build_attr_dataset,
    decode_generated_activities,
    deltas_from_timestamps,
    encode_case,
    load_temporal_cases,
)
from utils.dataset import pad_mask_from_ids, pad_sequences_1d
from utils.metrics import suffix_score
from model.model import PaCT, set_seed

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "benchmark"
RESULT_DIR = ROOT_DIR / "result"

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
# Columns the CSV reports. Numeric metric columns are formatted as floats; the
# identity/count columns are handled explicitly. Summary (mean/std) rows blank
# out per-fold-only columns (counts and timings).
METRIC_COLUMNS = [
    "next_activity_accuracy",
    "suffix_dl_similarity",
    "remaining_time_mae_days",
    "next_time_mae_days",
]
COUNT_COLUMNS = ["n_test_cases", "n_test_pairs"]
TIMING_COLUMNS = ["train_time_sec", "inference_time_sec"]
# Columns left blank on mean/std summary rows.
SUMMARY_BLANK_COLUMNS = set(COUNT_COLUMNS + TIMING_COLUMNS)

CSV_HEADERS = (
    ["dataset", "prefix_mode", "prefix_patch_size", "fold"]
    + METRIC_COLUMNS
    + COUNT_COLUMNS
    + TIMING_COLUMNS
)


VALID_PREFIX_MODES = ("wo_attr", "w_attr")


def _normalize_prefix_mode(prefix_mode: str) -> str:
    """Return the canonical prefix mode, accepting only the two supported keys.

    Prefix modes are produced internally and written to results.csv as either
    "wo_attr" (activity-only prefix) or "w_attr" (activity plus attributes).
    Empty strings are tolerated for partially written CSV rows.
    """
    if prefix_mode in VALID_PREFIX_MODES or prefix_mode == "":
        return prefix_mode
    raise ValueError(f"Unknown prefix_mode {prefix_mode!r}; expected one of {VALID_PREFIX_MODES}.")


def _cat_keys_for_dataset(dataset_name: str) -> List[str]:
    """Return configured categorical XES attributes for one benchmark file name."""
    config_key = FILENAME_TO_CONFIG_KEY.get(dataset_name)
    if config_key is None:
        return []
    return list(DATASET_CONFIG.get(config_key, []))


def _prefix_mode_label(prefix_mode: str) -> str:
    """Return the display label for an internal prefix mode key."""
    return _normalize_prefix_mode(prefix_mode)


def _resolve_run_mode(dataset_name: str, prefix_mode_override: str) -> Tuple[str, List[str]]:
    """Resolve the experiment prefix mode and categorical keys for one dataset."""
    cat_keys = _cat_keys_for_dataset(dataset_name)
    if prefix_mode_override == "auto":
        prefix_mode = "w_attr" if cat_keys else "wo_attr"
        return prefix_mode, cat_keys
    if prefix_mode_override == "wo_attr":
        return "wo_attr", []
    if prefix_mode_override == "w_attr":
        if not cat_keys:
            raise ValueError(f"{dataset_name} has no configured categorical attributes for w_attr mode.")
        return "w_attr", cat_keys
    raise ValueError(f"Unsupported prefix_mode_override: {prefix_mode_override}")


def _result_csv_path(dataset_name: str) -> Path:
    """Create and return the per-dataset experiment results.csv path."""
    dataset_dir = RESULT_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir / "results.csv"


def _checkpoint_path(dataset_name: str, prefix_mode: str, prefix_patch_size: int, fold_num: int) -> Path:
    """Create and return the checkpoint path for one dataset/mode/patch/fold."""
    prefix_mode = _normalize_prefix_mode(prefix_mode)
    ckpt_dir = RESULT_DIR / dataset_name / "checkpoints" / f"{prefix_mode}_w{int(prefix_patch_size)}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"fold_{int(fold_num)}.pt"


def _save_fold_checkpoint(dataset_name: str, prefix_mode: str, fold_num: int, model) -> Path:
    """Persist one trained fold with enough metadata to recover evaluation later."""
    prefix_mode = _normalize_prefix_mode(prefix_mode)
    if getattr(model, "_model", None) is None or getattr(model._model, "net", None) is None:
        raise ValueError("Cannot save checkpoint before the model is fitted.")
    pact_model = model._model
    prefix_patch_size = int(getattr(model, "prefix_patch_size", 1))
    ckpt_path = _checkpoint_path(dataset_name, prefix_mode, prefix_patch_size, fold_num)
    torch.save(
        {
            "variant": "stage1_intra_patch",
            "dataset": dataset_name,
            "prefix_mode": prefix_mode,
            "fold": int(fold_num),
            "prefix_patch_size": prefix_patch_size,
            "model_name": getattr(pact_model, "name", pact_model.__class__.__name__),
            "state_dict": pact_model.net.state_dict(),
            "attribute_dims": list(getattr(pact_model, "attribute_dims", []) or []),
            "eos_id": getattr(pact_model, "eos_id", None),
            "sos_id": getattr(pact_model, "sos_id", None),
            "encoders": getattr(model, "_encoders", None),
            "cat_keys": list(getattr(model, "cat_keys", []) or []),
            "max_steps": getattr(model, "_max_steps", None),
            "best_epoch": getattr(pact_model, "best_epoch_", 0),
            "n_epochs_run": getattr(pact_model, "n_epochs_run_", 0),
            "pact_config": {
                "prefix_patch_size": prefix_patch_size,
                "d_emb": pact_model.d_emb,
                "d_model": pact_model.d_model,
                "nhead": pact_model.nhead,
                "num_prefix_layers": pact_model.num_prefix_layers,
                "num_decoder_layers": pact_model.num_decoder_layers,
                "d_ff": pact_model.d_ff,
                "dropout": pact_model.dropout,
                "max_seq_len": pact_model.max_seq_len,
                "n_epochs": pact_model.n_epochs,
                "batch_size": pact_model.batch_size,
                "lr": pact_model.lr,
                "weight_decay": pact_model.weight_decay,
                "early_stopping_patience": pact_model.early_stopping_patience,
                "seed": pact_model.seed,
                "checkpoint_metric": "val_loss",
            },
        },
        ckpt_path,
    )
    return ckpt_path


def _iter_result_csvs() -> List[Path]:
    """List all per-dataset result CSV files under the PaCT result root."""
    if not RESULT_DIR.exists():
        return []
    return sorted(RESULT_DIR.glob("*/results.csv"))


# ---------------------------------------------------------------------------
# CSV value formatting
# ---------------------------------------------------------------------------
def _fmt_float(value: object) -> str:
    """Format numeric CSV values with six decimal places."""
    return f"{float(value):.6f}"


def _fmt_optional_float(value: object) -> str:
    """Format optional numeric CSV values, leaving missing metrics blank."""
    return "" if value is None else _fmt_float(value)


def _reduce_optional(reducer, values: list):
    """Apply reducer only to non-None values; return None if all values are None."""
    valid = [v for v in values if v is not None]
    return None if not valid else reducer(valid)


def _format_fold_result_row(
    dataset_name: str,
    prefix_mode: str,
    result: Dict,
    prefix_patch_size: int,
    fold: int,
) -> Dict[str, object]:
    """Convert one fold result dict into a CSV row with stable column names."""
    prefix_mode = _normalize_prefix_mode(prefix_mode)
    row: Dict[str, object] = {
        "dataset": dataset_name,
        "prefix_mode": prefix_mode,
        "prefix_patch_size": prefix_patch_size,
        "fold": fold,
    }
    for column in METRIC_COLUMNS:
        row[column] = _fmt_optional_float(result.get(column))
    for column in COUNT_COLUMNS:
        row[column] = result[column]
    for column in TIMING_COLUMNS:
        row[column] = _fmt_float(result[column])
    return row


def _format_summary_result_row(
    dataset_name: str,
    prefix_mode: str,
    fold_results: List[Dict],
    prefix_patch_size: int,
    fold: str,
) -> Dict[str, object]:
    """Build a mean or std CSV summary row from all fold result dicts."""
    prefix_mode = _normalize_prefix_mode(prefix_mode)
    reducer = np.mean if fold == "mean" else np.std
    row: Dict[str, object] = {
        "dataset": dataset_name,
        "prefix_mode": prefix_mode,
        "prefix_patch_size": prefix_patch_size,
        "fold": fold,
    }
    for column in METRIC_COLUMNS:
        values = [r.get(column) for r in fold_results]
        row[column] = _fmt_optional_float(_reduce_optional(reducer, values))
    for column in SUMMARY_BLANK_COLUMNS:
        row[column] = ""
    return row


def _result_key(dataset_name: str, prefix_mode: str, prefix_patch_size: int) -> Tuple[str, str, str]:
    """Return the normalized result key used for CSV/checkpoint resume."""
    return dataset_name, _normalize_prefix_mode(prefix_mode), str(int(prefix_patch_size))


def _row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    """Return the normalized result key for one CSV row."""
    return (
        row.get("dataset", ""),
        _normalize_prefix_mode(row.get("prefix_mode", "")),
        str(row.get("prefix_patch_size", "1") or "1"),
    )


def _csv_row_to_result(row: Dict[str, str]) -> Dict:
    """Convert one persisted fold CSV row back into a numeric fold result dict."""
    result: Dict[str, object] = {}
    for key in METRIC_COLUMNS + TIMING_COLUMNS:
        value = row.get(key, "")
        result[key] = None if value == "" else float(value)
    for key in COUNT_COLUMNS:
        value = row.get(key, "")
        result[key] = None if value == "" else int(float(value))
    return result


def _existing_fold_results(dataset_name: str, prefix_mode: str, prefix_patch_size: int) -> Dict[int, Dict]:
    """Load completed fold rows for a dataset/mode/patch result block."""
    result_csv = _result_csv_path(dataset_name)
    if not result_csv.exists():
        return {}
    key = _result_key(dataset_name, prefix_mode, prefix_patch_size)
    completed: Dict[int, Dict] = {}
    with result_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fold = row.get("fold", "")
            if _row_key(row) == key and fold.isdigit():
                completed[int(fold)] = _csv_row_to_result(row)
    return completed


def _save_resumable_dataset_results(
    dataset_name: str,
    prefix_mode: str,
    fold_results_by_num: Dict[int, Dict],
    prefix_patch_size: int,
    n_splits: int,
) -> None:
    """Merge available fold rows into results.csv and add summary when complete."""
    prefix_mode = _normalize_prefix_mode(prefix_mode)
    result_csv = _result_csv_path(dataset_name)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    key = _result_key(dataset_name, prefix_mode, prefix_patch_size)
    preserved: List[Dict[str, object]] = []

    if result_csv.exists():
        with result_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if _row_key(row) != key:
                    preserved.append(row)

    target_rows = [
        _format_fold_result_row(dataset_name, prefix_mode, fold_results_by_num[fold], prefix_patch_size, fold)
        for fold in sorted(fold_results_by_num)
    ]
    if len(fold_results_by_num) >= n_splits and all(fold in fold_results_by_num for fold in range(1, n_splits + 1)):
        ordered = [fold_results_by_num[fold] for fold in range(1, n_splits + 1)]
        target_rows.append(_format_summary_result_row(dataset_name, prefix_mode, ordered, prefix_patch_size, "mean"))
        target_rows.append(_format_summary_result_row(dataset_name, prefix_mode, ordered, prefix_patch_size, "std"))

    with result_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(preserved + target_rows)
    print(f"  CSV updated: {result_csv} ({len(fold_results_by_num)}/{n_splits} folds)")


def _load_fold_checkpoint(ckpt_path: Path, wrapper) -> Dict:
    """Load one fold checkpoint into an experiment wrapper for evaluation resume."""
    ckpt = torch.load(ckpt_path, map_location=wrapper._device, weights_only=False)
    pact_config = dict(ckpt["pact_config"])
    pact_config.pop("checkpoint_metric", None)
    pact_config.pop("suffix_patch_size", None)
    pact_config.pop("use_temporal", None)
    pact_config["prefix_patch_size"] = int(ckpt["prefix_patch_size"])
    model = PaCT(**pact_config)

    class _DatasetInfo:
        attribute_dims = ckpt["attribute_dims"]
        eos_id = ckpt["eos_id"]
        sos_id = ckpt["sos_id"]
        encoders = ckpt["encoders"]

    model._build(_DatasetInfo())
    assert model.net is not None
    model.net.load_state_dict(ckpt["state_dict"], strict=False)
    model.net.eval()
    model.best_epoch_ = int(ckpt.get("best_epoch", 0))
    model.n_epochs_run_ = int(ckpt.get("n_epochs_run", 0))
    model.train_time_per_epoch_ = 0.0
    model.val_time_sec_ = 0.0

    wrapper._model = model
    wrapper._encoders = ckpt["encoders"]
    wrapper._eos_id = int(ckpt["eos_id"])
    wrapper._sos_id = int(ckpt["sos_id"])
    wrapper._max_steps = int(ckpt.get("max_steps") or 200)
    return ckpt


def _assemble_fold_result(
    suffix_metrics: Dict[str, object],
    test_cases: list,
    train_time_sec: float,
    inference_time_sec: float,
) -> Dict:
    """Assemble a fold result dict from evaluated suffix metrics and timing info."""
    return {
        "next_activity_accuracy": float(suffix_metrics["next_activity_accuracy"]),
        "suffix_dl_similarity": float(suffix_metrics["suffix_dl_similarity"]),
        "next_time_mae_days": suffix_metrics.get("next_time_mae_days"),
        "remaining_time_mae_days": suffix_metrics.get("remaining_time_mae_days"),
        "n_test_cases": len(test_cases),
        "n_test_pairs": suffix_metrics["n_test_pairs"],
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
    }


def _compute_next_activity_accuracy(pred_suffix: List[str], true_suffix: List[str]) -> float:
    """Compute next-activity prediction accuracy from the first suffix token."""
    if not pred_suffix or not true_suffix:
        return 0.0
    return float(pred_suffix[0] == true_suffix[0])


class PaCTWrapper:
    """Benchmark adapter for PaCT with categorical and timestamp prefixes."""

    def __init__(
        self,
        cat_keys: List[str],
        val_ratio: float = 0.2,
        seed: int = 42,
        **pact_kwargs,
    ):
        """Store categorical settings and initialize model state placeholders."""
        self.cat_keys = cat_keys
        self.val_ratio = val_ratio
        self.seed = seed
        self.pact_kwargs = pact_kwargs
        self._model: Optional[PaCT] = None
        self._encoders: Optional[Dict] = None
        self._eos_id: Optional[int] = None
        self._sos_id: Optional[int] = None
        self._max_steps = 200
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def prefix_patch_size(self) -> int:
        """Expose the configured prefix patch size for result reporting."""
        return self.pact_kwargs.get("prefix_patch_size", 1)

    def fit(self, train_cases: List[List[Dict[str, str]]]) -> None:
        """Split event-dict cases, build datasets, and fit a PaCT model."""
        set_seed(self.seed)
        n_val = max(1, int(len(train_cases) * self.val_ratio))
        idx = np.random.default_rng(self.seed).permutation(len(train_cases))
        tr_cases = [train_cases[i] for i in idx[n_val:]]
        vl_cases = [train_cases[i] for i in idx[:n_val]]
        self._max_steps = max((len(case) for case in train_cases), default=200)

        train_ds = _build_attr_dataset(tr_cases, self.cat_keys)
        self._encoders = train_ds.encoders
        self._eos_id = train_ds.eos_id
        self._sos_id = train_ds.sos_id
        val_ds = _build_attr_dataset(vl_cases, self.cat_keys, encoders=self._encoders)

        self._model = PaCT(seed=self.seed, **self.pact_kwargs)
        self._model.fit(train_ds, val_dataset=val_ds)
        del train_ds, val_ds
        gc.collect()

    def rollout_suffix_batch(
        self,
        prefixes: List[List[Dict[str, str]]],
        batch_size: int = 128,
    ) -> Tuple[List[List[str]], List[Optional[List[float]]], List[Optional[float]]]:
        """Roll out suffix activity and temporal predictions for event-dict prefixes."""
        assert self._model is not None and self._model.net is not None
        assert self._encoders is not None and self._eos_id is not None and self._sos_id is not None
        classes = self._encoders["act"].classes_
        delta_norm = self._encoders["delta_time_norm"]
        results: List[Optional[List[str]]] = [None] * len(prefixes)
        next_results: List[Optional[List[float]]] = [None] * len(prefixes)
        remaining_results: List[Optional[float]] = [None] * len(prefixes)
        valid_idx: List[int] = []
        valid_act: List[List[int]] = []
        valid_cat: List[List[List[int]]] = []
        valid_time: List[List[float]] = []

        for idx, prefix in enumerate(prefixes):
            clean = [event for event in prefix if event["name"] != EOC_TOKEN]
            if not clean:
                results[idx] = [EOC_TOKEN]
                next_results[idx] = []
                remaining_results[idx] = None
                continue
            encoded = encode_case(clean, self._encoders, self.cat_keys)
            valid_idx.append(idx)
            valid_act.append(encoded.act_ids)
            valid_cat.append(encoded.cat_ids)
            valid_time.append([delta_norm.transform(delta) for delta in encoded.deltas])

        self._model.net.eval()
        for start in range(0, len(valid_idx), batch_size):
            chunk_act = valid_act[start:start + batch_size]
            chunk_cat = valid_cat[start:start + batch_size]
            chunk_time = valid_time[start:start + batch_size]
            chunk_idx = valid_idx[start:start + batch_size]
            max_len = max(len(ids) for ids in chunk_act)

            prefix_act = pad_sequences_1d(
                [torch.tensor(ids, dtype=torch.long) for ids in chunk_act], max_len, torch.long
            )
            prefix_pad_mask = pad_mask_from_ids(prefix_act)
            prefix_cat = [
                pad_sequences_1d(
                    [torch.tensor(cat_ids_by_key[key_idx], dtype=torch.long) for cat_ids_by_key in chunk_cat],
                    max_len,
                    torch.long,
                )
                for key_idx in range(len(self.cat_keys))
            ]
            prefix_time = pad_sequences_1d(
                [torch.tensor(times, dtype=torch.float) for times in chunk_time], max_len, torch.float, pad_val=0.0
            )

            with torch.no_grad():
                gen_act, next_time, remaining_time = self._model.net.rollout(
                    prefix_act=prefix_act.to(self._device),
                    prefix_cat=[tensor.to(self._device) for tensor in prefix_cat],
                    prefix_pad_mask=prefix_pad_mask.to(self._device),
                    eos_id=self._eos_id,
                    sos_id=self._sos_id,
                    max_steps=self._max_steps,
                    prefix_time=prefix_time.to(self._device),
                    return_temporal=True,
                )

            for row, original_idx in enumerate(chunk_idx):
                results[original_idx] = decode_generated_activities(gen_act[row].cpu().tolist(), classes, self._eos_id, self._sos_id)
                if next_time is not None:
                    next_results[original_idx] = next_time[row].detach().cpu().float().tolist()
                else:
                    next_results[original_idx] = []
                if remaining_time is not None:
                    remaining_results[original_idx] = float(remaining_time[row].detach().cpu().item())

        return (
            [result if result is not None else [EOC_TOKEN] for result in results],
            [result if result is not None else [] for result in next_results],
            remaining_results,
        )


class Evaluator:
    """Run k-fold cross-validation evaluation for one dataset and prefix mode."""

    def __init__(
        self,
        xes_path: Path,
        prefix_mode: str,
        cat_keys: Optional[List[str]] = None,
        n_splits: int = 5,
        random_state: int = 42,
        min_prefix_len: int = 1,
        eval_batch_size: int = 64,
    ):
        """Store dataset path and KFold/evaluation options for one experiment run."""
        self.xes_path = Path(xes_path)
        self.prefix_mode = prefix_mode
        self.cat_keys = cat_keys or []
        self.n_splits = n_splits
        self.random_state = random_state
        self.min_prefix_len = min_prefix_len
        self.eval_batch_size = int(eval_batch_size)

    def _load_cases(self):
        """Load cases in the representation required by this evaluator mode."""
        return load_temporal_cases(self.xes_path, self.cat_keys)

    def _iter_pair_chunks(self, cases, chunk_size: int):
        """Yield prefix/suffix evaluation pairs in bounded chunks."""
        prefixes = []
        suffixes = []
        suffix_next_targets = []
        remaining_targets = []
        for case in cases:
            clean_case = [event for event in case if event["name"] != EOC_TOKEN]
            timestamps = [float(event.get("_timestamp", 0.0)) for event in clean_case]
            deltas = deltas_from_timestamps(timestamps)
            end_ts = timestamps[-1] if timestamps else 0.0
            for prefix_len in range(self.min_prefix_len, len(case)):
                prefixes.append(case[:prefix_len])
                suffixes.append([event["name"] for event in case[prefix_len:]])
                suffix_next_targets.append(deltas[prefix_len:] + [0.0])
                remaining_targets.append(max(0.0, end_ts - timestamps[prefix_len - 1]) if timestamps else 0.0)
                if len(prefixes) >= chunk_size:
                    yield prefixes, suffixes, suffix_next_targets, remaining_targets
                    prefixes = []
                    suffixes = []
                    suffix_next_targets = []
                    remaining_targets = []
        if prefixes:
            yield prefixes, suffixes, suffix_next_targets, remaining_targets

    def _evaluate_suffix_stream(self, model, test_cases) -> Tuple[Dict[str, object], float]:
        """Compute suffix and temporal metrics with one streaming rollout pass."""
        next_act_sum = 0.0
        suffix_sum = 0.0
        n_pairs = 0
        next_abs_seconds_sum = 0.0
        next_abs_seconds_count = 0
        remaining_abs_seconds_sum = 0.0
        remaining_abs_seconds_count = 0
        delta_norm = None
        remaining_norm = None
        if getattr(model, "_encoders", None) is not None:
            delta_norm = model._encoders.get("delta_time_norm")
            remaining_norm = model._encoders.get("remaining_time_norm")

        infer_start = time.perf_counter()
        for prefixes, suffixes, next_targets, remaining_targets in self._iter_pair_chunks(
            test_cases,
            self.eval_batch_size,
        ):
            preds, next_preds, remaining_preds = model.rollout_suffix_batch(prefixes, batch_size=self.eval_batch_size)
            for pred, true_suffix, next_pred, next_target, remaining_pred, remaining_target in zip(
                preds,
                suffixes,
                next_preds,
                next_targets,
                remaining_preds,
                remaining_targets,
            ):
                next_act_sum += _compute_next_activity_accuracy(pred, true_suffix)
                suffix_sum += suffix_score(pred, true_suffix)
                n_pairs += 1
                if delta_norm is not None:
                    compare_len = min(len(next_pred or []), len(next_target), len(true_suffix))
                    for pos in range(compare_len):
                        if true_suffix[pos] == EOC_TOKEN:
                            continue
                        pred_z = float(next_pred[pos])
                        true_z = float(delta_norm.transform(next_target[pos]))
                        pred_seconds = float(delta_norm.inverse([pred_z])[0])
                        true_seconds = float(delta_norm.inverse([true_z])[0])
                        next_abs_seconds_sum += abs(pred_seconds - true_seconds)
                        next_abs_seconds_count += 1
                if remaining_norm is not None and remaining_pred is not None:
                    pred_z = float(remaining_pred)
                    true_z = float(remaining_norm.transform(remaining_target))
                    pred_seconds = float(remaining_norm.inverse([pred_z])[0])
                    true_seconds = float(remaining_norm.inverse([true_z])[0])
                    remaining_abs_seconds_sum += abs(pred_seconds - true_seconds)
                    remaining_abs_seconds_count += 1
            del prefixes, suffixes, next_targets, remaining_targets, preds, next_preds, remaining_preds
        inference_time_sec = time.perf_counter() - infer_start

        metrics = {
            "next_activity_accuracy": next_act_sum / n_pairs if n_pairs else 0.0,
            "suffix_dl_similarity": suffix_sum / n_pairs if n_pairs else 0.0,
            "n_test_pairs": n_pairs,
            "next_time_mae_days": (
                next_abs_seconds_sum / next_abs_seconds_count / SECONDS_PER_DAY
                if next_abs_seconds_count
                else None
            ),
            "remaining_time_mae_days": (
                remaining_abs_seconds_sum / remaining_abs_seconds_count / SECONDS_PER_DAY
                if remaining_abs_seconds_count
                else None
            ),
        }
        return metrics, inference_time_sec

    def _run_fold(self, model, train_cases, test_cases, fold_num: int) -> Dict:
        """Train one fold, predict all test prefixes, and return fold metrics."""
        print(f"  Fold {fold_num} | train={len(train_cases)} | test={len(test_cases)}")
        train_start = time.perf_counter()
        model.fit(train_cases)
        train_time_sec = time.perf_counter() - train_start
        dataset_name = self.xes_path.stem.replace(".xes", "")
        ckpt_path = _save_fold_checkpoint(dataset_name, self.prefix_mode, fold_num, model)
        print(f"  Fold {fold_num} | checkpoint saved: {ckpt_path}")
        if hasattr(model, "_max_steps"):
            model._max_steps = max((len(case) for case in train_cases + test_cases), default=model._max_steps)
        expected_pairs = sum(max(0, len(case) - self.min_prefix_len) for case in test_cases)
        print(f"  Fold {fold_num} | evaluating {expected_pairs} prefix-suffix pairs")

        suffix_metrics, inference_time_sec = self._evaluate_suffix_stream(model, test_cases)
        next_time_mae_days = suffix_metrics.get("next_time_mae_days")
        remaining_time_mae_days = suffix_metrics.get("remaining_time_mae_days")
        print(
            f"  Fold {fold_num} | "
            f"next_activity_accuracy={suffix_metrics['next_activity_accuracy']:.4f} "
            f"suffix_dl_similarity={suffix_metrics['suffix_dl_similarity']:.4f} "
            f"train={train_time_sec:.1f}s infer={inference_time_sec:.1f}s"
        )
        if next_time_mae_days is not None or remaining_time_mae_days is not None:
            print(
                f"  Fold {fold_num} | "
                f"next_time_mae_days={next_time_mae_days if next_time_mae_days is None else f'{next_time_mae_days:.4f}'} "
                f"remaining_time_mae_days={remaining_time_mae_days if remaining_time_mae_days is None else f'{remaining_time_mae_days:.4f}'}"
            )
        return _assemble_fold_result(
            suffix_metrics=suffix_metrics,
            test_cases=test_cases,
            train_time_sec=train_time_sec,
            inference_time_sec=inference_time_sec,
        )

    def _run_checkpoint_fold(self, model, test_cases, fold_num: int, ckpt_path: Path) -> Dict:
        """Evaluate one fold from an existing checkpoint without retraining."""
        print(f"  Fold {fold_num} | checkpoint found, evaluating only: {ckpt_path}")
        _load_fold_checkpoint(ckpt_path, model)
        if hasattr(model, "_max_steps"):
            model._max_steps = max((len(case) for case in test_cases), default=model._max_steps)

        expected_pairs = sum(max(0, len(case) - self.min_prefix_len) for case in test_cases)
        print(f"  Fold {fold_num} | evaluating {expected_pairs} prefix-suffix pairs")
        suffix_metrics, inference_time_sec = self._evaluate_suffix_stream(model, test_cases)
        result = _assemble_fold_result(
            suffix_metrics=suffix_metrics,
            test_cases=test_cases,
            train_time_sec=0.0,
            inference_time_sec=inference_time_sec,
        )
        print(
            f"  Fold {fold_num} | "
            f"next_activity_accuracy={result['next_activity_accuracy']:.4f} "
            f"suffix_dl_similarity={result['suffix_dl_similarity']:.4f} "
            f"infer={inference_time_sec:.1f}s"
        )
        return result

    def run(self, model) -> Dict:
        """Run all KFold splits, save CSV rows, and return dataset summary metrics."""
        print(f"\n{'=' * 60}")
        print(f"Dataset: {self.xes_path.name} | prefix_mode={_prefix_mode_label(self.prefix_mode)}")
        if self.cat_keys:
            print(f"Prefix attributes: {self.cat_keys}")
        print("Loading cases...")
        cases = self._load_cases()
        print(f"Cases: {len(cases)} | Events with EOC: {sum(len(case) for case in cases)}")

        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        dataset_name = self.xes_path.stem.replace(".xes", "")
        prefix_patch_size = getattr(model, "prefix_patch_size", 1)
        prefix_mode = _normalize_prefix_mode(self.prefix_mode)
        fold_results_by_num = _existing_fold_results(dataset_name, prefix_mode, prefix_patch_size)
        if fold_results_by_num:
            done_folds = ", ".join(str(fold) for fold in sorted(fold_results_by_num))
            print(f"Existing CSV folds for patch={prefix_patch_size}: {done_folds}")

        for fold_idx, (tr_idx, te_idx) in enumerate(kfold.split(np.arange(len(cases)))):
            fold_num = fold_idx + 1
            if fold_num in fold_results_by_num:
                print(f"  Fold {fold_num} | CSV row exists, skipping")
                continue

            ckpt_path = _checkpoint_path(dataset_name, prefix_mode, prefix_patch_size, fold_num)
            test_cases = [cases[i] for i in te_idx]
            if ckpt_path.exists():
                result = self._run_checkpoint_fold(model, test_cases, fold_num, ckpt_path)
            else:
                result = self._run_fold(
                    model,
                    [cases[i] for i in tr_idx],
                    test_cases,
                    fold_num=fold_num,
                )
            fold_results_by_num[fold_num] = result
            _save_resumable_dataset_results(
                dataset_name,
                self.prefix_mode,
                fold_results_by_num,
                prefix_patch_size,
                self.n_splits,
            )
            del test_cases
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all(fold in fold_results_by_num for fold in range(1, self.n_splits + 1)):
            missing = [fold for fold in range(1, self.n_splits + 1) if fold not in fold_results_by_num]
            raise RuntimeError(f"Missing fold results for {dataset_name} patch={prefix_patch_size}: {missing}")

        fold_results = [fold_results_by_num[fold] for fold in range(1, self.n_splits + 1)]
        _save_resumable_dataset_results(dataset_name, self.prefix_mode, fold_results_by_num, prefix_patch_size, self.n_splits)

        next_act_vals = [result["next_activity_accuracy"] for result in fold_results]
        suffix_vals = [result["suffix_dl_similarity"] for result in fold_results]
        next_time_vals = [r["next_time_mae_days"] for r in fold_results if r.get("next_time_mae_days") is not None]
        remaining_time_vals = [r["remaining_time_mae_days"] for r in fold_results if r.get("remaining_time_mae_days") is not None]
        summary = {
            "dataset": dataset_name,
            "prefix_mode": self.prefix_mode,
            "n_cases": len(cases),
            "folds": fold_results,
            "next_activity_accuracy_mean": float(np.mean(next_act_vals)),
            "next_activity_accuracy_std": float(np.std(next_act_vals)),
            "suffix_dl_similarity_mean": float(np.mean(suffix_vals)),
            "suffix_dl_similarity_std": float(np.std(suffix_vals)),
            "next_time_mean": float(np.mean(next_time_vals)) if next_time_vals else None,
            "next_time_std": float(np.std(next_time_vals)) if next_time_vals else None,
            "remaining_time_mean": float(np.mean(remaining_time_vals)) if remaining_time_vals else None,
            "remaining_time_std": float(np.std(remaining_time_vals)) if remaining_time_vals else None,
        }
        time_str = ""
        if summary["next_time_mean"] is not None:
            time_str += f" next_time={summary['next_time_mean']:.4f}+/-{summary['next_time_std']:.4f}d"
        if summary["remaining_time_mean"] is not None:
            time_str += f" remaining_time={summary['remaining_time_mean']:.4f}+/-{summary['remaining_time_std']:.4f}d"
        print(
            f"  Result | next_activity_accuracy={summary['next_activity_accuracy_mean']:.4f}"
            f"+/-{summary['next_activity_accuracy_std']:.4f} "
            f"suffix_dl_similarity={summary['suffix_dl_similarity_mean']:.4f}"
            f"+/-{summary['suffix_dl_similarity_std']:.4f}"
            + time_str
        )
        return summary


def _completed_datasets(n_splits: int = 5) -> set:
    """Return completed dataset/mode/patch keys from existing CSV rows."""
    counts: Dict[tuple, int] = {}
    for result_csv in _iter_result_csvs():
        with result_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("fold", "") in ("mean", "std", ""):
                    continue
                key = (
                    row["dataset"],
                    _normalize_prefix_mode(row.get("prefix_mode", "")),
                    row.get("prefix_patch_size", "1") or "1",
                )
                counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= n_splits}


def _select_datasets(available: Dict[str, Path], datasets: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """Resolve requested dataset names against available benchmark log files."""
    if datasets is None:
        ordered = [(name, available[name]) for name in DATASET_ORDER if name in available]
        extras = [(name, path) for name, path in available.items() if name not in set(DATASET_ORDER)]
        return ordered + sorted(extras)

    missing = [name for name in datasets if name not in available]
    if missing:
        print(f"[Warning] Missing dataset files: {missing}")
    order_map = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    valid = [(name, available[name]) for name in datasets if name in available]
    return sorted(valid, key=lambda item: order_map.get(item[0], 999))


def run_experiment(
    datasets: Optional[List[str]] = None,
    data_dir: Path = DATA_DIR,
    n_splits: int = 5,
    random_state: int = 42,
    min_prefix_len: int = 1,
    val_ratio: float = 0.2,
    seed: int = 42,
    prefix_patch_size: int = 1,
    eval_batch_size: int = 64,
    prefix_mode_override: str = "auto",
) -> Dict[str, Dict]:
    """Run the PaCT experiment over selected datasets.

    Inputs mirror the CLI flags: dataset selection, KFold settings, and patch
    size. Validation always uses val_loss for checkpoint selection.
    The function prints planned work, skips completed result blocks,
    trains/evaluates each remaining dataset, and finally prints a summary from
    all result CSVs.
    """
    if int(prefix_patch_size) != 1 and int(prefix_patch_size) % 2 != 0:
        raise ValueError("prefix_patch_size should be 1 or an even number.")
    if prefix_mode_override not in {"auto", "w_attr", "wo_attr"}:
        raise ValueError("prefix_mode_override must be one of: auto, w_attr, wo_attr")
    available = {path.name: path for path in data_dir.glob("*.xes*")}
    selected = _select_datasets(available, datasets)
    done = _completed_datasets(n_splits)
    prefix_patch_size_str = str(prefix_patch_size)

    print(
        f"\nRun order ({len(selected)} datasets, prefix_patch_size={prefix_patch_size_str}"
        f", prefix_mode_override={prefix_mode_override}, temporal, checkpoint_metric=val_loss):"
    )
    for idx, (name, _) in enumerate(selected, 1):
        stem = Path(name).stem.replace(".xes", "")
        result_prefix_mode, cat_keys = _resolve_run_mode(name, prefix_mode_override)
        key = (stem, result_prefix_mode, prefix_patch_size_str)
        status = "SKIP" if key in done else "TODO"
        attr_msg = f" attrs={cat_keys}" if cat_keys else ""
        print(f"  {idx:>2}. [{status}] {name} mode={_prefix_mode_label(result_prefix_mode)}{attr_msg}")
    print(f"CSV root: {RESULT_DIR}\n")

    all_results = {}
    for name, path in selected:
        stem = Path(name).stem.replace(".xes", "")
        result_prefix_mode, cat_keys = _resolve_run_mode(name, prefix_mode_override)
        key = (stem, result_prefix_mode, prefix_patch_size_str)
        if key in done:
            print(
                f"[SKIP] {name} prefix_mode={_prefix_mode_label(result_prefix_mode)} "
                f"prefix_patch_size={prefix_patch_size_str} already completed"
            )
            continue

        model = PaCTWrapper(
            cat_keys=cat_keys,
            val_ratio=val_ratio,
            seed=seed,
            prefix_patch_size=prefix_patch_size,
        )
        evaluator = Evaluator(
            path,
            prefix_mode=result_prefix_mode,
            cat_keys=cat_keys,
            n_splits=n_splits,
            random_state=random_state,
            min_prefix_len=min_prefix_len,
            eval_batch_size=eval_batch_size,
        )
        all_results[name] = evaluator.run(model)
        del model, evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("FINAL SUMMARY (PaCT main)")
    print("=" * 78)
    print(
        f"{'Dataset':<38} {'Mode':>8} {'P':>4} "
        f"{'next_act':>10} {'suffix':>10} {'next_t(d)':>10} {'rem_t(d)':>10}"
    )
    print("-" * 104)
    for result_csv in _iter_result_csvs():
        with result_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("fold") != "mean":
                    continue
                next_time = row.get("next_time_mae_days", "")
                remaining_time = row.get("remaining_time_mae_days", "")
                print(
                    f"{row['dataset']:<38} {_prefix_mode_label(row.get('prefix_mode', '')):>8} "
                    f"{row.get('prefix_patch_size', ''):>4} "
                    f"{float(row.get('next_activity_accuracy', 0.0)):>10.4f} "
                    f"{float(row.get('suffix_dl_similarity', 0.0)):>10.4f} "
                    f"{(float(next_time) if next_time else 0.0):>10.4f} "
                    f"{(float(remaining_time) if remaining_time else 0.0):>10.4f}"
                )
    return all_results