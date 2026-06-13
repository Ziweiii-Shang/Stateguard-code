"""
run_turn_level_gpt41.py — Turn-level probing analysis using GPT-4.1.

Replicates the turn-level analysis from run_all.py (run_turn_level_analysis)
but uses GPT-4.1 instead of GPT-4o to show the error accumulation effect
generalizes across models.

Usage: cd D:\experiment && python -u run_turn_level_gpt41.py
"""
import json
import os
import sys
import time

import numpy as np

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from run_all import probe_one_dialogue, bootstrap_ci, load_all_data


def run_turn_level_analysis_gpt41(all_data: dict) -> dict:
    """Stratify probing accuracy by turn position and input type using GPT-4.1.

    This replicates run_turn_level_analysis from run_all.py but with
    model='gpt-4.1' instead of 'gpt-4o', to demonstrate that the error
    accumulation and adversarial fragility effects generalize across models.
    """
    print("\n" + "=" * 60)
    print("Turn-Level Analysis (GPT-4.1) for §3.3")
    print("=" * 60)

    model = "gpt-4.1"
    procbench = all_data["procbench"]

    # Get common vs rare vs adversarial dialogues
    common_dlgs = [d for d in procbench if d.get("variant") == "common"][:25]
    adv_dlgs = [d for d in procbench if d.get("variant") == "adversarial"][:25]
    rare_dlgs = [d for d in procbench if d.get("variant") == "rare"][:25]

    print(f"  Dialogues: {len(common_dlgs)} common, {len(rare_dlgs)} rare, {len(adv_dlgs)} adversarial")

    results = {
        "by_position": {"early": [], "middle": [], "late": []},
        "by_position_rare": {"early": [], "middle": [], "late": []},
        "by_variant": {"common": [], "rare": [], "adversarial": []},
    }

    # --- Common dialogues by turn position ---
    print("  Probing common dialogues by turn position...")
    sys.stdout.flush()
    for i, d in enumerate(common_dlgs):
        r = probe_one_dialogue(model, d)
        for pt in r["per_turn"]:
            idx = pt["turn_idx"]
            if idx <= 3:
                pos = "early"
            elif idx <= 7:
                pos = "middle"
            else:
                pos = "late"
            results["by_position"][pos].append(1.0 if pt["correct"] else 0.0)
        if r["total"] > 0:
            results["by_variant"]["common"].append(r["correct"] / r["total"])
        if (i + 1) % 5 == 0:
            print(f"    Common: {i+1}/{len(common_dlgs)} done")
            sys.stdout.flush()

    # --- Rare dialogues by turn position ---
    print("  Probing rare dialogues by turn position...")
    sys.stdout.flush()
    for i, d in enumerate(rare_dlgs):
        r = probe_one_dialogue(model, d)
        for pt in r["per_turn"]:
            idx = pt["turn_idx"]
            if idx <= 3:
                pos = "early"
            elif idx <= 7:
                pos = "middle"
            else:
                pos = "late"
            results["by_position_rare"][pos].append(1.0 if pt["correct"] else 0.0)
        if r["total"] > 0:
            results["by_variant"]["rare"].append(r["correct"] / r["total"])
        if (i + 1) % 5 == 0:
            print(f"    Rare: {i+1}/{len(rare_dlgs)} done")
            sys.stdout.flush()

    # --- Adversarial dialogues ---
    print("  Probing adversarial dialogues...")
    sys.stdout.flush()
    for i, d in enumerate(adv_dlgs):
        r = probe_one_dialogue(model, d)
        if r["total"] > 0:
            results["by_variant"]["adversarial"].append(r["correct"] / r["total"])
        if (i + 1) % 5 == 0:
            print(f"    Adversarial: {i+1}/{len(adv_dlgs)} done")
            sys.stdout.flush()

    # --- Compute summary statistics ---
    summary = {"model": "gpt-4.1"}

    for key in ["by_position", "by_position_rare"]:
        summary[key] = {}
        for pos, vals in results[key].items():
            if vals:
                summary[key][pos] = {
                    "accuracy": round(np.mean(vals) * 100, 1),
                    "ci": bootstrap_ci([v * 100 for v in vals]),
                    "n": len(vals),
                }
            else:
                summary[key][pos] = {"accuracy": 0, "ci": (0, 0), "n": 0}

    summary["by_variant"] = {}
    for var, vals in results["by_variant"].items():
        if vals:
            summary["by_variant"][var] = {
                "accuracy": round(np.mean(vals) * 100, 1),
                "ci": bootstrap_ci([v * 100 for v in vals]),
                "n": len(vals),
            }
        else:
            summary["by_variant"][var] = {"accuracy": 0, "ci": (0, 0), "n": 0}

    # --- Compute key deltas for paper claims ---
    common_early = summary["by_position"]["early"]["accuracy"]
    common_late = summary["by_position"]["late"]["accuracy"]
    rare_early = summary["by_position_rare"]["early"]["accuracy"]
    rare_late = summary["by_position_rare"]["late"]["accuracy"]

    summary["error_accumulation"] = {
        "common_early_late_drop": round(common_early - common_late, 1),
        "rare_early_late_drop": round(rare_early - rare_late, 1),
    }

    adv_acc = summary["by_variant"]["adversarial"]["accuracy"]
    common_acc = summary["by_variant"]["common"]["accuracy"]
    summary["adversarial_fragility"] = {
        "common_accuracy": common_acc,
        "adversarial_accuracy": adv_acc,
        "drop": round(common_acc - adv_acc, 1),
    }

    return summary


def print_summary(summary: dict, gpt4o_results: dict = None):
    """Print a formatted summary of the results, optionally comparing to GPT-4o."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (GPT-4.1)")
    print("=" * 60)

    print("\n  --- By Position (Common Dialogues) ---")
    for pos in ["early", "middle", "late"]:
        data = summary["by_position"][pos]
        line = f"    {pos:>6}: {data['accuracy']:5.1f}% CI={data['ci']}  n={data['n']}"
        if gpt4o_results and pos in gpt4o_results.get("by_position", {}):
            g4o = gpt4o_results["by_position"][pos]["accuracy"]
            line += f"  (GPT-4o: {g4o}%)"
        print(line)

    print("\n  --- By Position (Rare Dialogues) ---")
    for pos in ["early", "middle", "late"]:
        data = summary["by_position_rare"][pos]
        line = f"    {pos:>6}: {data['accuracy']:5.1f}% CI={data['ci']}  n={data['n']}"
        if gpt4o_results and pos in gpt4o_results.get("by_position_rare", {}):
            g4o = gpt4o_results["by_position_rare"][pos]["accuracy"]
            line += f"  (GPT-4o: {g4o}%)"
        print(line)

    print("\n  --- By Variant ---")
    for var in ["common", "rare", "adversarial"]:
        data = summary["by_variant"][var]
        line = f"    {var:>12}: {data['accuracy']:5.1f}% CI={data['ci']}  n={data['n']}"
        if gpt4o_results and var in gpt4o_results.get("by_variant", {}):
            g4o = gpt4o_results["by_variant"][var]["accuracy"]
            line += f"  (GPT-4o: {g4o}%)"
        print(line)

    ea = summary["error_accumulation"]
    print(f"\n  --- Error Accumulation ---")
    print(f"    Common early->late drop: {ea['common_early_late_drop']}pp")
    print(f"    Rare   early->late drop: {ea['rare_early_late_drop']}pp")
    if gpt4o_results and "error_accumulation" in gpt4o_results:
        g4o_ea = gpt4o_results["error_accumulation"]
        print(f"    (GPT-4o common drop: {g4o_ea['common_early_late_drop']}pp, rare drop: {g4o_ea['rare_early_late_drop']}pp)")

    af = summary["adversarial_fragility"]
    print(f"\n  --- Adversarial Fragility ---")
    print(f"    Common->Adversarial drop: {af['drop']}pp ({af['common_accuracy']}% -> {af['adversarial_accuracy']}%)")
    if gpt4o_results and "adversarial_fragility" in gpt4o_results:
        g4o_af = gpt4o_results["adversarial_fragility"]
        print(f"    (GPT-4o drop: {g4o_af['drop']}pp ({g4o_af['common_accuracy']}% -> {g4o_af['adversarial_accuracy']}%))")


def main():
    start = time.time()
    print("=" * 60)
    print("GPT-4.1 Turn-Level Probing Analysis")
    print("=" * 60)

    output_path = os.path.join(RESULTS_DIR, "turn_level_gpt41_results.json")

    # Check if results already exist (from a previous run)
    if os.path.exists(output_path) and "--rerun" not in sys.argv:
        print(f"  [CACHE] Loading existing results from {output_path}")
        with open(output_path) as f:
            summary = json.load(f)
    else:
        # Load data and run analysis
        all_data = load_all_data()
        summary = run_turn_level_analysis_gpt41(all_data)

        # Save results
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results saved to {output_path}")

    # Load GPT-4o results for comparison
    gpt4o_path = os.path.join(RESULTS_DIR, "turn_level_results.json")
    gpt4o_results = None
    if os.path.exists(gpt4o_path):
        with open(gpt4o_path) as f:
            gpt4o_results = json.load(f)

    # Print summary
    print_summary(summary, gpt4o_results)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
