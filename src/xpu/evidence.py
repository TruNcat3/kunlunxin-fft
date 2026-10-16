"""Evidence-completeness verdicts shared by the selector builder and the
cost-model validation records (issue #3 review: one rule set, one verdict).

A focus candidate may only influence a promotion comparison or a model
metric when its evidence is *complete*: it exists in every declared
session, no session recorded a failure (``extras_failed``), the artifact
marks the candidate complete, the timing and error are finite and
correctness-gated, and the session/trial counts match the declared
protocol. Anything short of that is rejected with a reason — never a
silent skip. ``scripts/build_selector_table.py`` and
``src.xpu.cost_model.records_from_summary`` both call
:func:`focus_candidate_verdict`, so the builder's promotion gates and the
validator's record set can never drift apart.
"""

from __future__ import annotations

import math


def _finite_positive(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def focus_candidate_verdict(shape: dict, name: str, *, threshold: float,
                            expected_sessions: int, trials: int,
                            label: str = "") -> tuple[float | None, str | None]:
    """Complete-evidence verdict for one candidate in one focus shape.

    Returns ``(median_ms, None)`` when the candidate may be compared, else
    ``(None, reason)``. Rejects: absent candidate, a recorded session
    failure, a missing/False ``complete`` flag, a non-finite or
    non-positive timing, a missing/non-finite/wrong-value error, an error
    at or above the correctness threshold, and session or trial counts
    short of the declared protocol.
    """
    where = f"{label} {name}" if label else name
    cand = (shape.get("candidates") or {}).get(name)
    if cand is None:
        return None, (f"{where}: absent from the session set "
                      "(incomplete evidence; cannot promote)")
    failed = {f.get("candidate")
              for f in (shape.get("extras_failed") or [])}
    if name in failed:
        return None, (f"{where}: recorded failure in a session "
                      "(extras_failed); cannot promote")
    if cand.get("complete") is not True:
        return None, (f"{where}: complete evidence flag not true "
                      f"({cand.get('complete')!r}); cannot promote")
    ms = cand.get("median_ms")
    if not _finite_positive(ms):
        return None, f"{where}: median not finite positive ({ms!r})"
    err = cand.get("err_max", cand.get("err"))
    if not isinstance(err, (int, float)) or not math.isfinite(err):
        return None, f"{where}: missing/non-finite error ({err!r})"
    if err >= threshold:
        return None, (f"{where}: max_rel_error {err:.3e} >= {threshold:g}")
    got_s = cand.get("sessions")
    if got_s != expected_sessions:
        return None, (f"{where}: session evidence {got_s!r}/"
                      f"{expected_sessions} incomplete; cannot promote")
    want_trials = expected_sessions * trials
    n_trials = cand.get("n_trials")
    if n_trials != want_trials:
        return None, (f"{where}: trial evidence {n_trials!r}/{want_trials} "
                      "incomplete; cannot promote")
    return ms, None
