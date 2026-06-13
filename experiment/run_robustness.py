"""Run robustness analysis only (Table 3). Saves robustness_results.json."""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from run_all import (load_all_data, run_baseline_prompted, run_baseline_react_rules,
                     run_baseline_nemo, run_stateguard, evaluate_system,
                     N_ROBUSTNESS_COMMON, N_ROBUSTNESS_RARE, N_ROBUSTNESS_ADV)

def main():
    all_data = load_all_data()
    pfsas_path = os.path.join(RESULTS_DIR, "pfsas.json")
    with open(pfsas_path) as f:
        pfsas = json.load(f)

    procbench = all_data["procbench"]
    pfsa = pfsas.get("procbench", {})

    subsets = {
        "common": [d for d in procbench if d.get("variant") == "common"][:N_ROBUSTNESS_COMMON],
        "rare": [d for d in procbench if d.get("variant") == "rare"][:N_ROBUSTNESS_RARE],
        "adversarial": [d for d in procbench if d.get("variant") == "adversarial"][:N_ROBUSTNESS_ADV],
    }

    systems = {
        "prompted": lambda d: run_baseline_prompted(d),
        "react_rules": lambda d: run_baseline_react_rules(d),
        "nemo": lambda d: run_baseline_nemo(d),
        "stateguard": lambda d: run_stateguard(d, pfsa),
    }

    results = {}
    for sys_name, run_fn in systems.items():
        results[sys_name] = {}
        for subset_name, subset_dlgs in subsets.items():
            print(f"  {sys_name} / {subset_name} ({len(subset_dlgs)} dialogues)...")
            sys.stdout.flush()
            evaluated = [run_fn(d) for d in subset_dlgs]
            metrics = evaluate_system(evaluated)
            results[sys_name][subset_name] = {
                "PCR": metrics["PCR"], "PCR_ci": metrics["PCR_ci"],
                "SVR": metrics["SVR"], "n": metrics["n"],
            }
            print(f"    PCR = {metrics['PCR']}% [{metrics['PCR_ci']}]")
            sys.stdout.flush()
            # Save incrementally
            with open(os.path.join(RESULTS_DIR, "robustness_results.json"), "w") as f:
                json.dump(results, f, indent=2)

    print("ROBUSTNESS COMPLETE")

if __name__ == "__main__":
    main()
