"""Run model-agnostic analysis only (Table 5). Saves model_agnostic_results.json."""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from run_all import (load_all_data, run_baseline_prompted, run_stateguard,
                     evaluate_system, N_MODEL_AGNOSTIC)

def main():
    all_data = load_all_data()
    pfsas_path = os.path.join(RESULTS_DIR, "pfsas.json")
    with open(pfsas_path) as f:
        pfsas = json.load(f)

    procbench = all_data["procbench"]
    pfsa = pfsas.get("procbench", {})

    common = [d for d in procbench if d.get("variant") == "common"]
    adv = [d for d in procbench if d.get("variant") == "adversarial"]
    test_set = common[:N_MODEL_AGNOSTIC//2] + adv[:N_MODEL_AGNOSTIC//2]
    random.shuffle(test_set)
    test_set = test_set[:N_MODEL_AGNOSTIC]

    results = {}
    for model in AGNOSTIC_MODELS:
        print(f"  Backend: {model} ({len(test_set)} dialogues)...")
        sys.stdout.flush()

        base_evaluated = [run_baseline_prompted(d, model=model) for d in test_set]
        base_metrics = evaluate_system(base_evaluated)

        sg_evaluated = [run_stateguard(d, pfsa, model=model) for d in test_set]
        sg_metrics = evaluate_system(sg_evaluated)

        results[model] = {
            "base_PCR": base_metrics["PCR"], "base_PCR_ci": base_metrics["PCR_ci"],
            "sg_PCR": sg_metrics["PCR"], "sg_PCR_ci": sg_metrics["PCR_ci"],
            "improvement": round(sg_metrics["PCR"] - base_metrics["PCR"], 1),
            "base_TSR": base_metrics["TSR"], "sg_TSR": sg_metrics["TSR"],
        }
        print(f"    Base PCR={base_metrics['PCR']}%, +SG PCR={sg_metrics['PCR']}%, D={results[model]['improvement']}pp")
        sys.stdout.flush()
        # Save incrementally
        with open(os.path.join(RESULTS_DIR, "model_agnostic_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("MODEL-AGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()
