"""Run ablation analysis only (Table 4). Saves ablation_results.json."""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from run_all import (load_all_data, run_baseline_prompted, run_stateguard,
                     evaluate_system, N_ABLATION)

def main():
    all_data = load_all_data()
    pfsas_path = os.path.join(RESULTS_DIR, "pfsas.json")
    with open(pfsas_path) as f:
        pfsas = json.load(f)

    procbench = all_data["procbench"]
    pfsa = pfsas.get("procbench", {})

    # KEY: use adversarial+rare only
    hard_dialogues = [d for d in procbench if d.get("variant") in ("adversarial", "rare")]
    random.shuffle(hard_dialogues)
    hard_dialogues = hard_dialogues[:N_ABLATION]
    print(f"  Using {len(hard_dialogues)} adversarial+rare dialogues for ablation")

    # Random PFSA
    random_pfsa = json.loads(json.dumps(pfsa))
    if "transition_matrices" in random_pfsa:
        for k in random_pfsa["transition_matrices"]:
            mat = random_pfsa["transition_matrices"][k]
            n = len(mat)
            random_pfsa["transition_matrices"][k] = [[1.0/n]*n for _ in range(n)]

    variants = {
        "full": lambda d: run_stateguard(d, pfsa),
        "no_pfsa": lambda d: run_baseline_prompted(d),
        "hard_only": lambda d: run_stateguard(d, pfsa, use_graded=False),
        "no_importance": lambda d: run_stateguard(d, pfsa, use_importance=False),
        "no_smoothing": lambda d: run_stateguard(d, pfsa, use_smoothing=False),
        "random_pfsa": lambda d: run_stateguard(d, random_pfsa),
    }

    results = {}
    for var_name, run_fn in variants.items():
        print(f"  Ablation: {var_name} ({len(hard_dialogues)} dialogues)...")
        sys.stdout.flush()
        evaluated = [run_fn(d) for d in hard_dialogues]
        metrics = evaluate_system(evaluated)
        results[var_name] = {
            "PCR": metrics["PCR"], "PCR_ci": metrics["PCR_ci"],
            "SVR": metrics["SVR"], "TSR": metrics["TSR"], "n": metrics["n"],
        }
        print(f"    PCR = {metrics['PCR']}% [{metrics['PCR_ci']}]")
        sys.stdout.flush()
        # Save incrementally
        with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("ABLATION COMPLETE")

if __name__ == "__main__":
    main()
