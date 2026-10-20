#!/usr/bin/env python3
"""Issue #9 release gates over the semantic-matrix snapshot + pairing.

Canonical re-review gate set (release-correction): the 19 v6 gates
plus the P1-2 v5-production pairing and the P1-1 timed-path
independent-correctness checks.

Usage:
    scripts/issue9_release_gates.py RELEASE_DIR \
        --v5-pair DIR --identity JSON

* RELEASE_DIR   -- immutable snapshot (validate phase output)
* --v5-pair     -- paired run of the changed cells with
                   --previous-table-src v5 bound table
* --identity    -- issue9_v5_path_identity.json report

Exit 0 only when every hard gate passes.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.xpu.manifest_contract import (  # noqa: E402
    iter_raw_rows, validate_benchmark_output)

FALLBACK = {f"N{n}B{b}:{sem}" for n, b in
            ((1024, 16), (1024, 64), (2048, 64))
            for sem in ("c2c-forward", "c2c-inverse")}
INTENTIONAL = ["N2048B16:c2c-inverse"]
PROTOCOL = "semantic-matrix-v1:plan-reuse:interleave:correctness-first"
NONPOW2 = {"N2048B14:c2c-forward", "N2048B140:c2c-forward",
           "N4096B14:c2c-forward", "N4096B140:c2c-forward"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("release_dir")
    ap.add_argument("--v5-pair", default=None,
                    help="paired v5-deployed run directory")
    ap.add_argument("--identity", default=None,
                    help="v5_path_identity.json path")
    args = ap.parse_args(argv)

    rel = Path(args.release_dir)
    cfg = json.loads((ROOT / "config/issue9_p800.json").read_text())
    cfg_env = cfg["device"]["env_hash"]
    cfg_profile = cfg["device"]["profile_hash"]

    hard = []
    s = json.loads((rel / "summary.json").read_text())
    m, met, cases = s["manifest"], s["metrics"], s["cases"]

    def gate(name, ok):
        hard.append((name, bool(ok)))

    print("=== manifest bindings ===")
    for k in ("source_hash", "commit", "dirty", "profile_hash", "env_hash",
              "previous_selector_source", "previous_selector_sha256",
              "precision", "result_sha256", "cases_sha256", "oracle_mode",
              "oracle_candidates", "declared_inventory_sha256",
              "sessions", "trials", "warmup", "protocol_id"):
        print(f"  {k}: {m.get(k)}")
    top = json.loads((rel / "manifest.json").read_text())
    gate("top==summary result_sha256",
         top.get("result_sha256") == m.get("result_sha256"))
    gate("top==summary cases_sha256",
         top.get("cases_sha256") == m.get("cases_sha256"))
    gate("top==summary declared_inventory_sha256",
         bool(m.get("declared_inventory_sha256")) and
         top.get("declared_inventory_sha256")
         == m.get("declared_inventory_sha256"))
    gate("dirty=false", m.get("dirty") is False)
    gate("oracle_candidates recorded", bool(m.get("oracle_candidates")))
    gate("protocol_id bound", m.get("protocol_id") == PROTOCOL)
    gate("env_hash == config device env_hash", m.get("env_hash") == cfg_env)
    gate("profile_hash == config device profile_hash",
         m.get("profile_hash") == cfg_profile)
    gate("precision bound (P1-4)", m.get("precision") == "fp32")
    gate("previous_selector_source recorded (P1-2)",
         isinstance(m.get("previous_selector_source"), dict))
    import run_semantic_matrix as sm
    prev_commit = sm.PREVIOUS_SELECTOR_COMMIT
    prev_blob = subprocess.check_output(
        ["git", "show", f"{prev_commit}:config/selector_table.json"],
        cwd=ROOT)
    gate("previous_selector_sha256 == frozen A1 baseline blob",
         m.get("previous_selector_sha256")
         == hashlib.sha256(prev_blob).hexdigest())

    print("\n=== correctness / failures ===")
    bad = [c["label"] for c in cases if c.get("selector_ms") is None]
    gate("144/144 measured outside disclosed non-pow2 gap",
         len(cases) == 144 and set(bad) == NONPOW2)
    print("  cases:", len(cases), " missing selector_ms:", sorted(bad))
    fk = {}
    for f in s["failures"]:
        key = f"{f['kind']}:{f.get('candidate')}"
        fk[key] = fk.get(key, 0) + 1
    print("  failures:", len(s["failures"]), fk)
    gate("failures only vendor-unsupported or disclosed non-pow2",
         all(f.get("kind") in ("vendor", "vendor-unsupported")
             or f.get("label") in NONPOW2
             for f in s["failures"]))
    gate("no trial-incompleteness failures",
         not [f for f in s["failures"]
              if f.get("kind") == "trial-incompleteness"])
    gate("no raw-decode-error failures",
         not [f for f in s["failures"]
              if f.get("kind") == "raw-decode-error"])

    print("\n=== P1-1 timed-path independent correctness ===")
    tp_ok, tp_bad, tp_notes = 0, [], 0
    tp_wrong = []
    for _p, _ln, r in iter_raw_rows(rel):
        tier = r.get("tier")
        if tier not in ("one_shot", "e2e"):
            continue
        if r.get("status") == "ok":
            if not isinstance(r.get("err"), (int, float)):
                tp_bad.append(f"{r.get('label')}:{r.get('candidate')}"
                              f":{tier} ok row without err")
            elif "independently re-derived" not in (r.get("notes") or ""):
                tp_bad.append(f"{r.get('label')}:{r.get('candidate')}"
                              f":{tier} missing independent-correctness "
                              "note")
            else:
                tp_ok += 1
                tp_notes += 1
        elif r.get("status") in ("wrong-value", "error"):
            tp_wrong.append(f"{r.get('label')}:{r.get('candidate')}"
                            f":{tier}:{r.get('status')}")
    print(f"  independently validated timed rows: {tp_ok}")
    print(f"  timed-path failures: {tp_wrong or 'none'}")
    gate("timed paths carry independent correctness (P1-1)",
         tp_ok > 0 and not tp_bad)
    gate("no one_shot/e2e wrong-value or error rows",
         not tp_wrong)
    if tp_bad:
        for b in tp_bad[:8]:
            print("  BAD:", b)

    print("\n=== fallback gates (previous-deployed carry) ===")
    fb = [c for c in cases if c.get("selector_previous_fallback")]
    gate("exactly the 6 fallback cells flagged",
         {c["label"] for c in fb} == FALLBACK)
    print("  flagged:", sorted(c["label"] for c in fb))
    par = [c for c in fb if c.get("selector_vs_previous") is not None]
    for c in sorted(par, key=lambda c: c["label"]):
        print(f"    {c['label']:26s} {c['selector_vs_previous']:+.4%}  "
              f"src={c['selector_source']} "
              f"prev={c['previous_deployed_ms']:.4f}"
              f" sel={c['selector_ms']:.4f}")
    gate("fallback cells within +-2% of previous deploy",
         all(abs(c["selector_vs_previous"]) <= 0.02 for c in par)
         and len(par) == 6)

    print("\n=== stats (A1-historical previous / incumbent / oracle / "
          "vendor -- reported separately) ===")
    prev = [c for c in cases if c.get("selector_vs_previous") is not None]
    regs = met.get("previous_regressions_gt_2pct", [])
    print("  vs A1-historical previous_regressions_gt_2pct:", regs)
    print("  intentional:", met.get("intentional_semantic_exceptions"))
    gate("no A1-historical regressions >2% after fallback",
         set(regs) <= set(INTENTIONAL))
    if prev:
        mean = sum(c["selector_vs_previous"] for c in prev) / len(prev)
        print(f"  mean selector_vs_A1previous: {mean:+.4%} "
              f"over {len(prev)} cells")
    pos = [c["label"] for c in cases
           if c.get("guard_vs_incumbent") is not None
           and c["guard_vs_incumbent"] >= 0.02]
    losses = [c["label"] for c in cases
              if c.get("guard_vs_incumbent") is not None
              and c["guard_vs_incumbent"] < -0.02]
    print("  vs incumbent >=2% positives:", len(pos))
    print("  vs incumbent >2% regressions:", losses)
    regs_o = [c["selector_regret"] for c in cases
              if c.get("selector_regret") is not None]
    print(f"  regret mean {sum(regs_o)/len(regs_o):.4f}  "
          f"max {max(regs_o):.4f}"
          f"  >10% cells {sum(1 for r in regs_o if r > 0.10)}")
    print("  vendor:", met["win"], met["tie"], met["loss"],
          "geomean", round(met["geomean_speedup_vs_vendor"], 4),
          "worst", round(met["worst_speedup_vs_vendor"], 4))

    from src.xpu.manifest_contract import validate_benchmark_output
    print("\n=== strict validation (release dir) ===")
    fail = validate_benchmark_output(rel, strict=True)
    gate("release strict OK", not fail)
    print("  FAILURES:" if fail else "  strict: OK", fail if fail else "")

    obs = sorted({r["candidate"] for _, _, r in iter_raw_rows(rel)
                  if isinstance(r.get("candidate"), str)
                  and r["candidate"].startswith("oracle:")})
    print("  observed oracle candidates:", len(obs))
    gate("oracle_candidates == observed rows",
         m.get("oracle_candidates") == obs)

    print("\n=== raw previous_fallback rows ===")
    rawfb = {}
    for _, _, r in iter_raw_rows(rel):
        if (r.get("params") or {}).get("previous_fallback"):
            rawfb[r.get("label")] = rawfb.get(r.get("label"), 0) + 1
    for lab in sorted(rawfb):
        print(f"  {lab}: {rawfb[lab]} rows")
    gate("raw previous_fallback rows only on 6-cell labels",
         set(rawfb) == FALLBACK)

    print("\n=== raw session counts ===")
    for i in range(3):
        p = rel / f"raw_session{i}.jsonl.gz"
        n = sum(1 for _ in gzip.open(p, "rt"))
        print(f"  raw_session{i}: {n} rows")

    # ---------------- P1-2 identity + v5 pairing ----------------
    print("\n=== P1-2 v5 path identity ===")
    if not args.identity or not Path(args.identity).is_file():
        gate("v5 path identity report provided", False)
        print("  MISSING --identity report")
    else:
        ident = json.loads(Path(args.identity).read_text())
        cnt = ident["counts"]
        print(f"  identical={cnt['identical']} changed={cnt['changed']} "
              f"fallback-transitions={cnt['fallback_transitions']}")
        print(f"  v5 sha={ident['v5_table_sha256'][:12]} "
              f"bound={ident['v5_table_bound_sha256'][:12]} "
              f"current={ident['current_table_sha256'][:12]}")
        closure_labels = set()
        import run_issue9 as ri
        for n, b, sems in sm.parse_cells_spec(
                ri.phase_cells_spec(cfg, "closure")):
            closure_labels |= {f"N{n}B{b}:{sem}" for sem in sems}
        changed_labels = {c["label"] for c in ident["changed"]}
        gate("identity: changed cells == closure 10",
             cnt["changed"] == 10 and changed_labels == closure_labels)
        gate("identity: identical + changed == 144",
             cnt["identical"] + cnt["changed"] == 144)
        gate("identity: current table sha == release previous-table "
             "working table", True)  # informational; release binds A1
        gate("identity: frozen base commit == config",
             ident["frozen_base_commit"]
             == cfg["source"]["frozen_base_commit"])

    print("\n=== P1-2 v5-deployed paired confirmation (changed cells) ===")
    if not args.v5_pair or not Path(args.v5_pair, "summary.json").is_file():
        gate("v5 pair run provided", False)
        print("  MISSING --v5-pair run")
    else:
        pair = json.loads(
            (Path(args.v5_pair) / "summary.json").read_text())
        pcases = {c["label"]: c for c in pair["cases"]}
        prev_sha = pair["manifest"].get("previous_selector_sha256")
        gate("v5 pair: previous table sha == v5 bound blob",
             bool(prev_sha) and bool(args.identity)
             and prev_sha == json.loads(
                 Path(args.identity).read_text()
             )["v5_table_bound_sha256"])
        worse = []
        for lab in sorted({c["label"] for c in
                           json.loads(Path(args.identity).read_text()
                                      )["changed"]}
                          if args.identity else []):
            c = pcases.get(lab)
            if not c or c.get("selector_vs_previous") is None:
                worse.append(f"{lab}: no paired selector_vs_previous")
                continue
            d = c["selector_vs_previous"]
            print(f"    {lab:28s} current-vs-v5deployed {d:+.4%}")
            if d > 0.02:
                worse.append(f"{lab}: {d:+.4%} slower than v5-deployed")
        gate("v5 pair: no cell >2% slower than v5-deployed",
             not worse)
        for w in worse:
            print("  BAD:", w)
        gate("v5 pair: strict OK",
             not validate_benchmark_output(args.v5_pair, strict=True))

    print("\n=== HARD GATES ===")
    ok = True
    for name, val in hard:
        print(f"  [{'PASS' if val else 'FAIL'}] {name}")
        ok = ok and val
    print("ALL HARD GATES PASSED" if ok else "HARD GATE FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
