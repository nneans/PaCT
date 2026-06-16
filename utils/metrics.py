"""Pure sequence metrics used by PaCT evaluation.

Keep this module independent from torch/model state. Model code can import
these functions, but metric implementations should not depend on PaCT modules.
"""
from __future__ import annotations

from typing import Sequence


def damerau_levenshtein(s1: Sequence[int], s2: Sequence[int]) -> int:
    """Return edit distance between two activity-id sequences.

    Input is two generated/target suffix sequences. The dynamic-programming
    table counts insertions, deletions, substitutions, and adjacent swaps, and
    the final bottom-right cell is used by suffix score normalization.
    """
    n, m = len(s1), len(s2)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if (
                i > 1 and j > 1
                and s1[i - 1] == s2[j - 2]
                and s1[i - 2] == s2[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[n][m]


def suffix_score(pred: Sequence[int], true: Sequence[int]) -> float:
    """Compute remaining-trace prefix similarity from two suffix sequences.

    Input is a predicted suffix and a true suffix after EOS/SOS cleanup. The
    Damerau-Levenshtein distance is divided by the longer sequence length and
    converted into a similarity score in which 1.0 means an exact match.
    """
    if not pred and not true:
        return 1.0
    return 1.0 - damerau_levenshtein(pred, true) / max(len(pred), len(true))