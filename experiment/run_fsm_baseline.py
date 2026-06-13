"""
run_fsm_baseline.py — Add hand-authored FSM baseline to experiments.

The FSM baseline uses the SOP specification directly (steps + mandatory_before)
to build a deterministic state machine monitor. NO distillation from LLM behavior.
This isolates the value of spectral learning distillation vs. direct SOP encoding.

Runs:
1. FSM baseline on main evaluation (Table 2) — all 3 datasets, n=40
2. FSM baseline on robustness subsets (Table 3) — ProcBench common/rare/adv
3. FSM baseline on ablation hard set (Table 4) — for comparison

Usage: cd d:/experiment && python -u run_fsm_baseline.py
"""
import json
import os
import sys
import random
import math
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from llm_utils import chat
from run_all import (
    load_all_data, load_cached_result, classify_action, evaluate_system,
    N_MAIN_EVAL, N_ROBUSTNESS_COMMON, N_ROBUSTNESS_RARE, N_ROBUSTNESS_ADV, N_ABLATION,
)


def run_baseline_fsm(dialogue: dict, model: str = BASELINE_MODEL) -> dict:
    """Run hand-authored FSM baseline.

    This baseline uses the SOP specification directly to build a deterministic
    finite-state machine:
    - States = SOP steps (ordered)
    - Transitions = step[i] -> step[i+1] (sequential)
    - Constraints = mandatory_before (hard-coded from SOP spec)
    - Guidance = deterministic: "next step is X", "cannot do Y until Z complete"

    Key difference from StateGuard: NO spectral learning, NO PFSA distillation,
    NO LLM behavior observation. Pure SOP-specification-driven monitoring.
    """
    sop_steps = dialogue.get("sop_steps", [])
    mandatory = dialogue.get("mandatory_before", {})
    steps_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sop_steps))
    mandatory_str = "\n".join(f"  - {k} requires: {', '.join(v)}" for k, v in mandatory.items())

    system_prompt = (
        f"You are a customer service agent. Follow this SoP:\n"
        f"Steps:\n{steps_str}\n"
    )

    new_turns = []
    messages = [{"role": "system", "content": system_prompt}]
    completed_steps = []
    # FSM state: index into sop_steps (deterministic, sequential)
    current_state_idx = 0

    for turn in dialogue.get("turns", []):
        if turn["speaker"] == "user":
            messages.append({"role": "user", "content": turn["text"]})
            new_turns.append(turn)
        elif turn["speaker"] == "agent":
            # FSM guidance: deterministic based on SOP spec
            remaining = [s for s in sop_steps if s not in completed_steps]
            next_expected = remaining[0] if remaining else sop_steps[-1]

            # Build guidance from FSM state
            guidance_parts = []
            guidance_parts.append(f"Current step: {next_expected} (step {sop_steps.index(next_expected)+1} of {len(sop_steps)}).")

            # Check prerequisites for upcoming steps
            blocked = []
            for step, prereqs in mandatory.items():
                if step not in completed_steps:
                    missing = [p for p in prereqs if p not in completed_steps]
                    if missing:
                        blocked.append(f"Cannot do {step} until {', '.join(missing)} completed.")

            if blocked:
                guidance_parts.append("CONSTRAINTS: " + " ".join(blocked))

            guidance = "\n[FSM Monitor] " + " ".join(guidance_parts)
            messages.append({"role": "system", "content": guidance})

            resp = chat(model, messages, max_tokens=200, temperature=0.3)
            action = classify_action(resp, sop_steps)

            # Hard violation check + regeneration (same as StateGuard)
            if action in mandatory:
                prereqs = mandatory[action]
                missing = [p for p in prereqs if p not in completed_steps]
                if missing:
                    correction = f"\n[FSM Monitor] VIOLATION: {action} blocked. Complete {', '.join(missing)} first. Next step should be: {next_expected}."
                    messages.append({"role": "system", "content": correction})
                    resp = chat(model, messages, max_tokens=200, temperature=0.3)
                    action = classify_action(resp, sop_steps)

            step_idx = sop_steps.index(action) if action in sop_steps else -1
            new_turns.append({"speaker": "agent", "text": resp, "action": action, "step_index": step_idx})
            messages.append({"role": "assistant", "content": resp})
            if action in sop_steps:
                completed_steps.append(action)
                # Advance FSM state
                if action == next_expected and current_state_idx < len(sop_steps) - 1:
                    current_state_idx += 1

    result = dialogue.copy()
    result["turns"] = new_turns
    return result


def main():
    start = time.time()
    print("=" * 60)
    print("FSM Baseline Experiment")
    print("=" * 60)

    all_data = load_all_data()

    # ---- Table 2: Main evaluation ----
    print("\n" + "=" * 60)
    print("FSM Baseline — Main Evaluation (Table 2)")
    print("=" * 60)

    main_results = {}
    for ds_name, dialogues in all_data.items():
        eval_dialogues = dialogues[:N_MAIN_EVAL]
        print(f"  {ds_name} / fsm ({len(eval_dialogues)} dialogues)...")
        sys.stdout.flush()
        evaluated = [run_baseline_fsm(d) for d in eval_dialogues]
        metrics = evaluate_system(evaluated)
        main_results[ds_name] = metrics
        print(f"    PCR={metrics['PCR']}% [{metrics['PCR_ci']}] SVR={metrics['SVR']} TSR={metrics['TSR']} Flu={metrics['Fluency']}")
        sys.stdout.flush()

    # ---- Table 3: Robustness ----
    print("\n" + "=" * 60)
    print("FSM Baseline — Robustness (Table 3)")
    print("=" * 60)

    procbench = all_data["procbench"]
    subsets = {
        "common": [d for d in procbench if d.get("variant") == "common"][:N_ROBUSTNESS_COMMON],
        "rare": [d for d in procbench if d.get("variant") == "rare"][:N_ROBUSTNESS_RARE],
        "adversarial": [d for d in procbench if d.get("variant") == "adversarial"][:N_ROBUSTNESS_ADV],
    }

    robustness_results = {}
    for subset_name, subset_dlgs in subsets.items():
        print(f"  fsm / {subset_name} ({len(subset_dlgs)} dialogues)...")
        sys.stdout.flush()
        evaluated = [run_baseline_fsm(d) for d in subset_dlgs]
        metrics = evaluate_system(evaluated)
        robustness_results[subset_name] = {
            "PCR": metrics["PCR"],
            "PCR_ci": metrics["PCR_ci"],
            "SVR": metrics["SVR"],
            "TSR": metrics["TSR"],
            "n": metrics["n"],
        }
        print(f"    PCR={metrics['PCR']}% [{metrics['PCR_ci']}] TSR={metrics['TSR']}%")
        sys.stdout.flush()

    # ---- Table 4: Ablation comparison ----
    print("\n" + "=" * 60)
    print("FSM Baseline — Ablation set (Table 4 comparison)")
    print("=" * 60)

    hard_dialogues = [d for d in procbench if d.get("variant") in ("adversarial", "rare")]
    random.shuffle(hard_dialogues)
    hard_dialogues = hard_dialogues[:N_ABLATION]
    print(f"  fsm / adversarial+rare ({len(hard_dialogues)} dialogues)...")
    sys.stdout.flush()
    evaluated = [run_baseline_fsm(d) for d in hard_dialogues]
    ablation_metrics = evaluate_system(evaluated)
    print(f"    PCR={ablation_metrics['PCR']}% TSR={ablation_metrics['TSR']}% SVR={ablation_metrics['SVR']}")

    # Save all results
    all_results = {
        "main": main_results,
        "robustness": robustness_results,
        "ablation": {
            "PCR": ablation_metrics["PCR"],
            "PCR_ci": ablation_metrics["PCR_ci"],
            "SVR": ablation_metrics["SVR"],
            "TSR": ablation_metrics["TSR"],
            "n": ablation_metrics["n"],
        },
    }

    results_path = os.path.join(RESULTS_DIR, "fsm_baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"FSM BASELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results saved to {results_path}")
    print(f"{'=' * 60}")

    # Summary
    print("\n=== FSM Main Results ===")
    for ds, m in main_results.items():
        print(f"  {ds}: PCR={m['PCR']}% SVR={m['SVR']} TSR={m['TSR']}% Flu={m['Fluency']}")
    print(f"\n=== FSM Robustness ===")
    for sub, m in robustness_results.items():
        print(f"  {sub}: PCR={m['PCR']}% TSR={m['TSR']}%")
    print(f"\n=== FSM Ablation set ===")
    print(f"  PCR={ablation_metrics['PCR']}% TSR={ablation_metrics['TSR']}%")


if __name__ == "__main__":
    main()
