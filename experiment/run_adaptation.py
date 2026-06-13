"""Run adaptation analysis only (Table 6). Saves adaptation_results.json."""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from run_all import (load_all_data, run_stateguard, evaluate_system,
                     extract_action_sequences, build_pfsa, N_ADAPTATION_TEST)

def main():
    all_data = load_all_data()
    procbench = all_data["procbench"]

    common = [d for d in procbench if d.get("variant") == "common"]
    adv = [d for d in procbench if d.get("variant") == "adversarial"]
    test_set = common[:N_ADAPTATION_TEST//2] + adv[:N_ADAPTATION_TEST//2]
    random.shuffle(test_set)
    test_set = test_set[:N_ADAPTATION_TEST]

    results = {}
    for n_examples in [20, 50, 100, 200]:
        print(f"  Distilling from {n_examples} examples, testing on {len(test_set)}...")
        sys.stdout.flush()
        train_set = procbench[:min(n_examples, len(procbench))]
        sequences, actions = extract_action_sequences(train_set)
        if not sequences:
            results[str(n_examples)] = {"PCR": 0.0, "PCR_ci": [0,0], "TSR": 0.0}
            continue

        pfsa = build_pfsa(sequences, actions)
        evaluated = [run_stateguard(d, pfsa) for d in test_set]
        metrics = evaluate_system(evaluated)
        results[str(n_examples)] = {
            "PCR": metrics["PCR"], "PCR_ci": metrics["PCR_ci"], "TSR": metrics["TSR"],
        }
        print(f"    PCR = {metrics['PCR']}% [{metrics['PCR_ci']}]")
        sys.stdout.flush()
        # Save incrementally
        with open(os.path.join(RESULTS_DIR, "adaptation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("ADAPTATION COMPLETE")

if __name__ == "__main__":
    main()
