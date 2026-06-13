"""
run_robustness_tsr.py — Collect TSR (Task Success Rate) for the 4 baseline
systems on the ProcBench robustness subsets (common, rare, adversarial).

The original robustness_results.json was saved without TSR.  This script
re-runs every system × subset combination using the SAME dialogues and
slice sizes (common[:30], rare[:20], adversarial[:20]), computes TSR via
check_task_success, and writes robustness_tsr_results.json.

Usage:  cd D:\experiment && python -u run_robustness_tsr.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR
from run_all import (
    load_all_data,
    load_cached_result,
    run_baseline_prompted,
    run_baseline_react_rules,
    run_baseline_nemo,
    run_stateguard,
    check_task_success,
)

# Same slice sizes as run_robustness() in run_all.py
N_ROBUSTNESS_COMMON = 30
N_ROBUSTNESS_RARE = 20
N_ROBUSTNESS_ADV = 20


def main():
    start = time.time()
    print("=" * 60)
    print("Robustness TSR Collection")
    print("=" * 60)

    # ---- data -------------------------------------------------------
    all_data = load_all_data()
    procbench = all_data["procbench"]

    # Load cached PFSA (needed by stateguard)
    pfsas = load_cached_result("pfsas")
    if pfsas is None:
        print("ERROR: pfsas.json not found in results/. Run run_all.py first.")
        sys.exit(1)
    pfsa = pfsas.get("procbench", {})

    # ---- build the exact same subsets as run_robustness() -----------
    subsets = {
        "common":       [d for d in procbench if d.get("variant") == "common"][:N_ROBUSTNESS_COMMON],
        "rare":         [d for d in procbench if d.get("variant") == "rare"][:N_ROBUSTNESS_RARE],
        "adversarial":  [d for d in procbench if d.get("variant") == "adversarial"][:N_ROBUSTNESS_ADV],
    }

    for subset_name, dlgs in subsets.items():
        print(f"  Subset '{subset_name}': {len(dlgs)} dialogues")

    # ---- systems ----------------------------------------------------
    systems = {
        "prompted":     lambda d: run_baseline_prompted(d),
        "react_rules":  lambda d: run_baseline_react_rules(d),
        "nemo":         lambda d: run_baseline_nemo(d),
        "stateguard":   lambda d: run_stateguard(d, pfsa),
    }

    # ---- run --------------------------------------------------------
    results = {}
    total_runs = 0

    for sys_name, run_fn in systems.items():
        results[sys_name] = {}
        for subset_name, subset_dlgs in subsets.items():
            n = len(subset_dlgs)
            print(f"\n  [{sys_name} / {subset_name}]  Running {n} dialogues ...")
            sys.stdout.flush()

            tsr_hits = 0
            for i, d in enumerate(subset_dlgs):
                evaluated = run_fn(d)
                if check_task_success(evaluated):
                    tsr_hits += 1
                total_runs += 1
                if (i + 1) % 5 == 0 or (i + 1) == n:
                    print(f"    ... {i+1}/{n}  (TSR so far: {tsr_hits}/{i+1})")
                    sys.stdout.flush()

            tsr_pct = round(tsr_hits / n * 100, 1) if n > 0 else 0.0
            results[sys_name][subset_name] = tsr_pct
            print(f"    => TSR = {tsr_pct}%  ({tsr_hits}/{n})")
            sys.stdout.flush()

    # ---- save -------------------------------------------------------
    out_path = os.path.join(RESULTS_DIR, "robustness_tsr_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ---- summary table ----------------------------------------------
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  TSR Summary  (elapsed {elapsed/60:.1f} min, {total_runs} system runs)")
    print(f"{'=' * 60}")
    header = f"  {'System':<15} {'Common':>8} {'Rare':>8} {'Adversarial':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for sys_name in ["prompted", "react_rules", "nemo", "stateguard"]:
        c = results[sys_name]["common"]
        r = results[sys_name]["rare"]
        a = results[sys_name]["adversarial"]
        print(f"  {sys_name:<15} {c:>7.1f}% {r:>7.1f}% {a:>11.1f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
