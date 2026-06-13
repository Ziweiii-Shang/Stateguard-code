"""demo_case.py — Re-run StateGuard with audit logging on the case-study
dialogues (fin_account_open #3/#4/#5, where the prompted baseline violated
KYC). Produces results/case_study_log.json for the paper's qualitative
section and appendix audit-trail listing. Does not touch main checkpoints.
"""
import json

from run_all import (load_all_data, run_stateguard, step_completion_rate,
                     check_compliance)

pfsas = json.load(open("results/pfsas.json"))
pb = load_all_data()["procbench"]

out = []
for idx in (3, 4, 5):
    d = run_stateguard(pb[idx], pfsas["procbench"])
    comp, viols, n, nc = check_compliance(d)
    tsr = step_completion_rate(d)
    n_block = sum(1 for e in d["audit_log"] if e["blocked_and_regenerated"])
    n_guided = sum(1 for e in d["audit_log"] if e["guidance_level"] != "none")
    print(f"#{idx} sop={d['sop_id']} violations={viols} TSR={tsr*100:.0f}% "
          f"guided_turns={n_guided} block_regen={n_block}", flush=True)
    out.append({"dialogue_idx": idx, "sop_id": d["sop_id"],
                "variant": d["variant"], "violations": viols,
                "tsr": round(tsr * 100, 2), "audit_log": d["audit_log"],
                "turns": d["turns"]})

with open("results/case_study_log.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved results/case_study_log.json")
