"""Winner-selection policy for tree self-consistency voting.

Default policy is plurality (one branch, one vote), unchanged. Setting
``AUTOTREE_VOTE=weighted`` switches to confidence-weighted voting: each answer
equivalence class is scored by ``sum(exp(beta * mean_logprob))`` over its
branches, so a smaller but more confident class can win. This targets measured
selection loss (a correct answer generated in some branch but out-voted by a
larger low-confidence class). ``AUTOTREE_VOTE_BETA`` (default 1.0) sets the
temperature. The weighted path engages only when every voting branch carries a
logprob; otherwise callers keep their plurality path.

The functions are pure and take pre-extracted (answer_key, logprob) pairs so
the scheduler runtime and the serving layer share one implementation.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Tuple


def vote_mode() -> str:
    return os.environ.get("AUTOTREE_VOTE", "plurality")


def vote_beta() -> float:
    raw = os.environ.get("AUTOTREE_VOTE_BETA", "1.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _group(items: Iterable[Tuple[Optional[str], Optional[float]]]) -> Dict[str, List[Optional[float]]]:
    groups: Dict[str, List[Optional[float]]] = {}
    for key, logprob in items:
        if key is None or key == "":
            continue
        groups.setdefault(key, []).append(logprob)
    return groups


def weighted_winning_key(
    items: Iterable[Tuple[Optional[str], Optional[float]]],
    beta: Optional[float] = None,
) -> Optional[str]:
    """Return the answer key of the highest confidence-weighted class, or None.

    Falls back to plurality (count, then best logprob) when any voting branch
    lacks a logprob, so the weighted path never silently drops confidence
    information it does not have."""
    groups = _group(items)
    if not groups:
        return None
    if beta is None:
        beta = vote_beta()
    have_all_logprobs = all(
        all(lp is not None for lp in lps) for lps in groups.values()
    )
    if have_all_logprobs:
        def mass(key: str) -> Tuple[float, float]:
            lps = groups[key]
            return (sum(math.exp(beta * lp) for lp in lps), max(lps))
        return max(groups, key=mass)
    # plurality fallback: count, tie-broken by the class's best available logprob
    def plurality(key: str) -> Tuple[int, float]:
        lps = [lp for lp in groups[key] if lp is not None]
        return (len(groups[key]), max(lps) if lps else float("-inf"))
    return max(groups, key=plurality)


def env_float(name: str, default: float) -> float:
    """Parse a float env var, treating unset OR empty string as the default.

    Guards the value-margin / threshold knobs: an env exported as an empty
    string used to reach float('') and crash server startup at import."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default
