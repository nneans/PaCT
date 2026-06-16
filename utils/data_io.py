"""XES log loading, event parsing, and temporal dataset building for PaCT.

This module owns all I/O that touches the event log format: reading .xes/.xes.gz
files, extracting timestamps, caching parsed cases, and converting raw cases
into PaCTDataset objects. Model training, evaluation, and CSV reporting live in
experiment.py and model/model.py.
"""
from __future__ import annotations

import gzip
import hashlib
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from utils.dataset import PaCTDataset, TimeNormalizer, fit_label_encoder, safe_transform_with_unk

EOC_TOKEN = "End of Case (EOC)"
SECONDS_PER_DAY = 86400.0
CACHE_DIR = Path(__file__).resolve().parent.parent / "result" / "cache" / "pact"


@dataclass
class EncodedCase:
    """One case encoded into model id sequences plus raw temporal values.

    act_ids and each list in cat_ids are 1-based, position-aligned, and the same
    length as the case; timestamps/deltas carry the raw (un-normalized) temporal
    signal so callers can apply prefix/suffix-specific time normalization.
    """
    act_ids: List[int]
    cat_ids: List[List[int]]
    timestamps: List[float]
    deltas: List[float]
    end_ts: float


def deltas_from_timestamps(timestamps: Sequence[float]) -> List[float]:
    """Return inter-event time deltas (seconds) for a list of event timestamps.

    The first event has delta 0.0; each later delta is clamped to be
    non-negative. Used by both dataset building and benchmark evaluation so the
    prefix/suffix delta features are computed identically everywhere.
    """
    deltas = [0.0]
    for idx in range(1, len(timestamps)):
        deltas.append(max(0.0, float(timestamps[idx]) - float(timestamps[idx - 1])))
    return deltas


def encode_case(
    case: Sequence[Dict[str, object]],
    encoders: Dict,
    cat_keys: Sequence[str],
) -> "EncodedCase":
    """Encode one case's event dicts into model id sequences and raw deltas.

    Activity and categorical labels are mapped to 1-based ids (0 is reserved for
    padding) via the fitted encoders, with unseen values folded into the UNK id.
    Timestamps yield per-event seconds, their inter-event deltas, and the case
    end time. This is the single source of truth for case encoding, shared by
    dataset building and benchmark rollout so both encode prefixes identically.
    """
    acts = [event["name"] for event in case]
    act_ids = [idx + 1 for idx in safe_transform_with_unk(encoders["act"], acts)]
    cat_ids = [
        [
            idx + 1
            for idx in safe_transform_with_unk(
                encoders[f"cat_{key}"],
                [event.get(key, "") for event in case],
            )
        ]
        for key in cat_keys
    ]
    timestamps = [float(event.get("_timestamp", 0.0)) for event in case]
    deltas = deltas_from_timestamps(timestamps)
    end_ts = timestamps[-1] if timestamps else 0.0
    return EncodedCase(
        act_ids=act_ids,
        cat_ids=cat_ids,
        timestamps=timestamps,
        deltas=deltas,
        end_ts=end_ts,
    )


def decode_generated_activities(
    tokens: Sequence[int],
    classes: Sequence,
    eos_id: int,
    sos_id: int,
) -> List[str]:
    """Decode generated activity ids back into names, ensuring EOC termination.

    This is the inverse of the activity encoding done by encode_case: ids are
    1-based over the fitted activity classes, so name = classes[id - 1]. PAD (0)
    and SOS tokens are skipped, EOS terminates the sequence, and a trailing EOC
    token is appended so every decoded suffix ends at case completion.
    """
    decoded: List[str] = []
    for token in tokens:
        if token == 0 or token == sos_id:
            continue
        if token == eos_id:
            decoded.append(EOC_TOKEN)
            break
        act_idx = token - 1
        if 0 <= act_idx < len(classes):
            decoded.append(str(classes[act_idx]))
    if not decoded or decoded[-1] != EOC_TOKEN:
        decoded.append(EOC_TOKEN)
    return decoded


def _load_xes_log(xes_path: Path):
    """Load a .xes or .xes.gz event log through pm4py.

    Gzipped logs are decompressed to a temporary .xes file because pm4py's XES
    importer expects a regular file path. The temporary file is removed after
    import and the pm4py log object is returned.
    """
    from pm4py.objects.log.importer.xes import importer as xes_importer

    path_str = str(xes_path)
    if path_str.endswith(".gz"):
        with gzip.open(path_str, "rb") as f_in:
            tmp = tempfile.NamedTemporaryFile(suffix=".xes", delete=False)
            shutil.copyfileobj(f_in, tmp)
            tmp.close()
        log = xes_importer.apply(tmp.name)
        Path(tmp.name).unlink(missing_ok=True)
        return log
    return xes_importer.apply(path_str)


def _event_timestamp_seconds(event) -> float:
    """Extract time:timestamp from a pm4py event as epoch seconds."""
    ts = event.get("time:timestamp")
    if ts is None:
        return 0.0
    if hasattr(ts, "timestamp"):
        return float(ts.timestamp())
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def load_temporal_cases(xes_path: "Path | str", cat_keys: List[str]) -> List[List[Dict[str, object]]]:
    """Load event dictionaries including timestamps for temporal PaCT.

    Input is a log path and categorical keys. Each event carries _timestamp
    seconds so dataset building can derive delta and remaining-time targets.
    """
    xes_path = Path(xes_path)
    stat = xes_path.stat()
    cache_key = hashlib.sha1(
        f"temporal|{xes_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{'|'.join(cat_keys)}".encode("utf-8")
    ).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    cases = []
    for trace in _load_xes_log(xes_path):
        events = []
        timestamps = []
        for event in trace:
            if "concept:name" not in event:
                continue
            ts = _event_timestamp_seconds(event)
            row: Dict[str, object] = {"name": str(event["concept:name"]), "_timestamp": ts}
            for key in cat_keys:
                row[key] = str(event[key]) if key in event else ""
            events.append(row)
            timestamps.append(ts)
        if events:
            events.append({"name": EOC_TOKEN, "_timestamp": timestamps[-1], **{key: "" for key in cat_keys}})
            cases.append(events)
    with cache_path.open("wb") as f:
        pickle.dump(cases, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cases


def _build_attr_dataset(
    cases: List[List[Dict[str, str]]],
    cat_keys: List[str],
    encoders: Optional[Dict] = None,
) -> PaCTDataset:
    """Convert activity+attribute cases into a PaCTDataset.

    Input cases are event dictionaries from load_temporal_cases. The function
    fits or reuses activity, categorical, and time normalizers, then emits one
    tensor sample per prefix length with activity suffix and temporal targets.
    """
    clean_cases = [[event for event in case if event["name"] != EOC_TOKEN] for case in cases]
    clean_cases = [case for case in clean_cases if case]
    if not clean_cases:
        raise ValueError("No valid cases after removing EOC tokens.")
    if any("_timestamp" not in event for case in clean_cases for event in case):
        raise ValueError("PaCT requires cases loaded with time:timestamp values.")

    if encoders is None:
        encoders = {"act": fit_label_encoder([event["name"] for case in clean_cases for event in case])}
        for key in cat_keys:
            encoders[f"cat_{key}"] = fit_label_encoder(
                [event.get(key, "") for case in clean_cases for event in case]
            )
        all_deltas = []
        all_remaining = []
        for case in clean_cases:
            timestamps = [float(event.get("_timestamp", 0.0)) for event in case]
            if not timestamps:
                continue
            deltas = deltas_from_timestamps(timestamps)
            deltas_with_eos = deltas + [0.0]
            all_deltas.extend(deltas_with_eos[1:-1])
            end_ts = timestamps[-1]
            all_remaining.extend(
                max(0.0, end_ts - timestamps[prefix_len - 1])
                for prefix_len in range(1, len(timestamps) + 1)
                if max(0.0, end_ts - timestamps[prefix_len - 1]) > 0.0
            )
        delta_norm = TimeNormalizer()
        delta_norm.fit(all_deltas)
        remaining_norm = TimeNormalizer()
        remaining_norm.fit(all_remaining)
        encoders["delta_time_norm"] = delta_norm
        encoders["remaining_time_norm"] = remaining_norm

    act_known = len(encoders["act"].classes_)
    eos_id = act_known + 2
    sos_id = act_known + 3
    attribute_dims = [sos_id] + [len(encoders[f"cat_{key}"].classes_) + 1 for key in cat_keys]
    delta_norm = encoders["delta_time_norm"]
    remaining_norm = encoders["remaining_time_norm"]

    samples = []
    for case_id, case in enumerate(clean_cases):
        encoded = encode_case(case, encoders, cat_keys)
        act_ids = encoded.act_ids
        deltas = encoded.deltas
        end_ts = encoded.end_ts
        timestamps = encoded.timestamps
        cat_ids_by_key = {key: encoded.cat_ids[k] for k, key in enumerate(cat_keys)}
        for prefix_len in range(1, len(act_ids) + 1):
            sample = {
                "case_id": case_id,
                "prefix_len": prefix_len,
                "prefix_act": torch.tensor(act_ids[:prefix_len], dtype=torch.long),
                "prefix_cat": [
                    torch.tensor(cat_ids_by_key[key][:prefix_len], dtype=torch.long)
                    for key in cat_keys
                ],
                "suffix_act": torch.tensor(act_ids[prefix_len:] + [eos_id], dtype=torch.long),
            }
            # The final entry is the EOS next-time placeholder. It is masked
            # out by the temporal loss/evaluation because EOS is not an event.
            suffix_next_raw = deltas[prefix_len:] + [0.0]
            remaining_raw = max(0.0, end_ts - timestamps[prefix_len - 1])
            sample.update({
                "prefix_time": torch.tensor(
                    [delta_norm.transform(delta) for delta in deltas[:prefix_len]],
                    dtype=torch.float,
                ),
                "suffix_next_time": torch.tensor(
                    [delta_norm.transform(delta) for delta in suffix_next_raw],
                    dtype=torch.float,
                ),
                "suffix_remaining_time": torch.tensor(remaining_norm.transform(remaining_raw), dtype=torch.float),
            })
            samples.append(sample)

    return PaCTDataset(
        samples=samples,
        attribute_dims=attribute_dims,
        eos_id=eos_id,
        sos_id=sos_id,
        encoders=encoders,
        cat_keys=cat_keys,
    )