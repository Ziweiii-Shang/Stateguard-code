"""
ProcBench: SoP definitions + dialogue generation + data loading.
Also handles MultiWOZ/SGD data extraction.
"""
import json
import os
import random
from typing import List, Dict, Tuple
from config import DATA_DIR, GEN_MODEL, N_PROCBENCH_PER_SOP
from llm_utils import chat, chat_json

# ============================================================
# 15 SoP Definitions (5 finance, 5 healthcare, 5 telecom)
# ============================================================
SOPS = [
    # --- Finance (5) ---
    {
        "id": "fin_account_open",
        "domain": "finance",
        "name": "Bank Account Opening",
        "steps": [
            "greet_customer",
            "collect_personal_info",
            "verify_identity",
            "select_account_type",
            "review_terms",
            "complete_application",
            "confirm_and_close",
        ],
        "branches": {
            "verify_identity": ["id_verified", "id_failed_retry", "id_failed_escalate"],
        },
        "mandatory_before": {"complete_application": ["verify_identity", "select_account_type"]},
    },
    {
        "id": "fin_loan_inquiry",
        "domain": "finance",
        "name": "Loan Inquiry and Pre-Approval",
        "steps": [
            "greet_customer",
            "identify_loan_type",
            "collect_financial_info",
            "run_credit_check",
            "present_options",
            "handle_questions",
            "schedule_followup",
            "close_conversation",
        ],
        "branches": {
            "run_credit_check": ["approved", "conditional", "denied"],
        },
        "mandatory_before": {"present_options": ["run_credit_check"]},
    },
    {
        "id": "fin_fraud_report",
        "domain": "finance",
        "name": "Fraud Report and Investigation",
        "steps": [
            "greet_customer",
            "verify_identity",
            "collect_incident_details",
            "freeze_account",
            "file_investigation",
            "provide_reference_number",
            "explain_next_steps",
            "close_conversation",
        ],
        "branches": {
            "freeze_account": ["frozen_success", "freeze_declined_review"],
        },
        "mandatory_before": {"freeze_account": ["verify_identity"], "file_investigation": ["collect_incident_details"]},
    },
    {
        "id": "fin_wire_transfer",
        "domain": "finance",
        "name": "International Wire Transfer",
        "steps": [
            "greet_customer",
            "verify_identity",
            "collect_recipient_info",
            "verify_amount_and_currency",
            "compliance_check",
            "confirm_transfer",
            "provide_confirmation_number",
            "close_conversation",
        ],
        "branches": {
            "compliance_check": ["passed", "flagged_for_review"],
        },
        "mandatory_before": {"confirm_transfer": ["verify_identity", "compliance_check"]},
    },
    {
        "id": "fin_dispute",
        "domain": "finance",
        "name": "Transaction Dispute",
        "steps": [
            "greet_customer",
            "verify_identity",
            "identify_transaction",
            "collect_dispute_reason",
            "initiate_chargeback",
            "issue_provisional_credit",
            "explain_timeline",
            "close_conversation",
        ],
        "branches": {
            "initiate_chargeback": ["eligible", "not_eligible_explain"],
        },
        "mandatory_before": {"initiate_chargeback": ["verify_identity", "identify_transaction"]},
    },
    # --- Healthcare (5) ---
    {
        "id": "health_triage",
        "domain": "healthcare",
        "name": "Symptom Triage",
        "steps": [
            "greet_patient",
            "verify_patient_identity",
            "collect_symptoms",
            "assess_severity",
            "check_medical_history",
            "provide_recommendation",
            "schedule_or_refer",
            "close_conversation",
        ],
        "branches": {
            "assess_severity": ["emergency_refer", "urgent_schedule", "routine_advice"],
        },
        "mandatory_before": {"provide_recommendation": ["collect_symptoms", "assess_severity"]},
    },
    {
        "id": "health_prescription",
        "domain": "healthcare",
        "name": "Prescription Refill",
        "steps": [
            "greet_patient",
            "verify_patient_identity",
            "identify_medication",
            "check_refill_eligibility",
            "verify_pharmacy",
            "process_refill",
            "confirm_pickup_details",
            "close_conversation",
        ],
        "branches": {
            "check_refill_eligibility": ["eligible", "needs_doctor_approval", "expired"],
        },
        "mandatory_before": {"process_refill": ["verify_patient_identity", "check_refill_eligibility"]},
    },
    {
        "id": "health_appointment",
        "domain": "healthcare",
        "name": "Appointment Scheduling",
        "steps": [
            "greet_patient",
            "verify_patient_identity",
            "determine_visit_type",
            "check_insurance",
            "find_available_slots",
            "confirm_appointment",
            "send_confirmation",
            "close_conversation",
        ],
        "branches": {
            "check_insurance": ["covered", "not_covered_discuss_options"],
        },
        "mandatory_before": {"confirm_appointment": ["verify_patient_identity", "check_insurance"]},
    },
    {
        "id": "health_lab_results",
        "domain": "healthcare",
        "name": "Lab Results Inquiry",
        "steps": [
            "greet_patient",
            "verify_patient_identity",
            "locate_lab_results",
            "explain_results",
            "answer_questions",
            "recommend_followup",
            "close_conversation",
        ],
        "branches": {
            "explain_results": ["normal_no_action", "abnormal_followup_needed"],
        },
        "mandatory_before": {"explain_results": ["verify_patient_identity", "locate_lab_results"]},
    },
    {
        "id": "health_referral",
        "domain": "healthcare",
        "name": "Specialist Referral",
        "steps": [
            "greet_patient",
            "verify_patient_identity",
            "review_reason_for_referral",
            "check_insurance_coverage",
            "identify_specialist",
            "submit_referral",
            "provide_specialist_info",
            "close_conversation",
        ],
        "branches": {
            "check_insurance_coverage": ["in_network", "out_of_network_options"],
        },
        "mandatory_before": {"submit_referral": ["verify_patient_identity", "check_insurance_coverage"]},
    },
    # --- Telecom (5) ---
    {
        "id": "tel_plan_change",
        "domain": "telecom",
        "name": "Plan Upgrade/Downgrade",
        "steps": [
            "greet_customer",
            "verify_account",
            "review_current_plan",
            "present_options",
            "handle_questions",
            "confirm_change",
            "process_change",
            "close_conversation",
        ],
        "branches": {
            "confirm_change": ["confirmed", "needs_more_time"],
        },
        "mandatory_before": {"process_change": ["verify_account", "confirm_change"]},
    },
    {
        "id": "tel_billing_dispute",
        "domain": "telecom",
        "name": "Billing Dispute",
        "steps": [
            "greet_customer",
            "verify_account",
            "identify_charge",
            "investigate_charge",
            "explain_finding",
            "apply_adjustment",
            "confirm_resolution",
            "close_conversation",
        ],
        "branches": {
            "explain_finding": ["charge_valid_explain", "charge_error_credit"],
        },
        "mandatory_before": {"apply_adjustment": ["verify_account", "investigate_charge"]},
    },
    {
        "id": "tel_tech_support",
        "domain": "telecom",
        "name": "Technical Support",
        "steps": [
            "greet_customer",
            "verify_account",
            "diagnose_issue",
            "attempt_remote_fix",
            "escalate_if_needed",
            "confirm_resolution",
            "close_conversation",
        ],
        "branches": {
            "attempt_remote_fix": ["fixed", "not_fixed_escalate"],
        },
        "mandatory_before": {"escalate_if_needed": ["diagnose_issue", "attempt_remote_fix"]},
    },
    {
        "id": "tel_cancellation",
        "domain": "telecom",
        "name": "Service Cancellation",
        "steps": [
            "greet_customer",
            "verify_account",
            "understand_reason",
            "offer_retention",
            "process_cancellation",
            "explain_final_bill",
            "confirm_cancellation",
            "close_conversation",
        ],
        "branches": {
            "offer_retention": ["accepted_stay", "declined_proceed"],
        },
        "mandatory_before": {"process_cancellation": ["verify_account", "understand_reason"]},
    },
    {
        "id": "tel_new_line",
        "domain": "telecom",
        "name": "Add New Line",
        "steps": [
            "greet_customer",
            "verify_account",
            "collect_new_user_info",
            "select_plan_for_new_line",
            "select_device",
            "review_order",
            "process_order",
            "close_conversation",
        ],
        "branches": {
            "select_device": ["device_in_stock", "device_backordered"],
        },
        "mandatory_before": {"process_order": ["verify_account", "review_order"]},
    },
]


def _generate_dialogue(sop: dict, variant: str, model: str = GEN_MODEL) -> dict:
    """Generate one dialogue following a SoP. variant in {common, rare, adversarial}."""
    steps_str = " -> ".join(sop["steps"])
    branch_str = json.dumps(sop.get("branches", {}))
    mandatory_str = json.dumps(sop.get("mandatory_before", {}))

    if variant == "common":
        user_instruction = "The user cooperates fully and follows the happy path. All verifications succeed on the first try."
    elif variant == "rare":
        user_instruction = (
            "The user hits a rare branch: a verification fails, an unusual condition triggers, "
            "or an edge case occurs. Include at least one non-happy-path branch."
        )
    else:  # adversarial
        user_instruction = (
            "The user tries to skip steps, asks to bypass verification, provides contradictory info, "
            "or attempts social engineering. The agent must resist and follow the SoP."
        )

    prompt = f"""Generate a realistic multi-turn customer service dialogue for: {sop['name']}

SoP steps: {steps_str}
Branches: {branch_str}
Mandatory prerequisites: {mandatory_str}

User behavior: {user_instruction}

Return JSON with this exact format:
{{
  "turns": [
    {{"speaker": "user"|"agent", "text": "...", "action": "<sop_step_name or none>", "step_index": <0-based index or -1>}}
  ]
}}

Rules:
- Each agent turn should map to exactly one SoP step (action field)
- User turns have action "none" and step_index -1
- Include 6-14 turns total (mix of user and agent)
- The agent MUST follow step ordering per mandatory_before
- For rare/adversarial: show the deviation AND the agent's correct handling"""

    result = chat_json(model, [{"role": "user", "content": prompt}], max_tokens=2048)
    if not result or "turns" not in result:
        # Fallback: create a minimal valid dialogue
        result = _fallback_dialogue(sop, variant)

    result["sop_id"] = sop["id"]
    result["variant"] = variant
    result["sop_steps"] = sop["steps"]
    result["mandatory_before"] = sop.get("mandatory_before", {})
    return result


def _fallback_dialogue(sop: dict, variant: str) -> dict:
    """Create a minimal valid dialogue if LLM generation fails."""
    turns = []
    for i, step in enumerate(sop["steps"]):
        turns.append({"speaker": "user", "text": f"I need help with step {i+1}.", "action": "none", "step_index": -1})
        turns.append({"speaker": "agent", "text": f"Processing {step}.", "action": step, "step_index": i})
    return {"turns": turns}


def generate_procbench(force: bool = False) -> List[dict]:
    """Generate all ProcBench dialogues. Caches to disk."""
    cache_path = os.path.join(DATA_DIR, "procbench.json")
    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            return json.load(f)

    print("[ProcBench] Generating dialogues...")
    all_dialogues = []
    for sop in SOPS:
        for variant, count in N_PROCBENCH_PER_SOP.items():
            for i in range(count):
                print(f"  {sop['id']} / {variant} / {i+1}/{count}", end="\r")
                dlg = _generate_dialogue(sop, variant)
                all_dialogues.append(dlg)

    with open(cache_path, "w") as f:
        json.dump(all_dialogues, f, indent=1)
    print(f"[ProcBench] Generated {len(all_dialogues)} dialogues.")
    return all_dialogues


# ============================================================
# MultiWOZ data loading (simplified: extract dialogues with process states)
# ============================================================
MULTIWOZ_BOOKING_STEPS = [
    "greet", "collect_domain", "collect_slots", "search_options",
    "present_options", "collect_booking_slots", "confirm_booking",
    "provide_reference", "close"
]


def load_multiwoz(n: int = 150) -> List[dict]:
    """Load MultiWOZ dialogues with process state annotations."""
    cache_path = os.path.join(DATA_DIR, "multiwoz_processed.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
            return data[:n]

    print("[MultiWOZ] Loading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("pfb30/multi_woz_v22", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"[MultiWOZ] Failed to load: {e}. Generating synthetic substitute.")
        return _generate_synthetic_tod("multiwoz", n)

    dialogues = []
    for item in ds:
        if len(dialogues) >= n * 2:  # load more, then filter
            break
        turns_raw = item.get("turns", {})
        utterances = turns_raw.get("utterance", [])
        speakers = turns_raw.get("speaker", [])
        if not utterances or len(utterances) < 4:
            continue

        turns = []
        step_idx = 0
        for j, (utt, spk) in enumerate(zip(utterances, speakers)):
            if spk == 1:  # system
                action = MULTIWOZ_BOOKING_STEPS[min(step_idx, len(MULTIWOZ_BOOKING_STEPS)-1)]
                turns.append({
                    "speaker": "agent", "text": utt,
                    "action": action, "step_index": min(step_idx, len(MULTIWOZ_BOOKING_STEPS)-1)
                })
                step_idx += 1
            else:
                turns.append({"speaker": "user", "text": utt, "action": "none", "step_index": -1})

        # Classify path frequency
        n_steps = step_idx
        variant = "common" if n_steps <= len(MULTIWOZ_BOOKING_STEPS) else "rare"

        dialogues.append({
            "turns": turns,
            "sop_id": "multiwoz_booking",
            "variant": variant,
            "sop_steps": MULTIWOZ_BOOKING_STEPS,
            "mandatory_before": {"confirm_booking": ["collect_slots", "search_options"]},
        })

    # Mark ~20% as rare
    for d in dialogues[int(len(dialogues)*0.8):]:
        d["variant"] = "rare"

    with open(cache_path, "w") as f:
        json.dump(dialogues, f, indent=1)
    print(f"[MultiWOZ] Loaded {len(dialogues)} dialogues.")
    return dialogues[:n]


def load_sgd(n: int = 150) -> List[dict]:
    """Load SGD dialogues with process state annotations."""
    cache_path = os.path.join(DATA_DIR, "sgd_processed.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
            return data[:n]

    print("[SGD] Loading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("schema_guided_dstc8", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"[SGD] Failed to load: {e}. Generating synthetic substitute.")
        return _generate_synthetic_tod("sgd", n)

    SGD_STEPS = [
        "greet", "identify_service", "collect_parameters", "call_api",
        "present_results", "collect_more_info", "confirm_action",
        "provide_details", "close"
    ]

    dialogues = []
    for item in ds:
        if len(dialogues) >= n * 2:
            break
        turns_data = item.get("turns", {})
        utterances = turns_data.get("utterance", [])
        speakers = turns_data.get("speaker", [])
        if not utterances or len(utterances) < 4:
            continue

        turns = []
        step_idx = 0
        for utt, spk in zip(utterances, speakers):
            if spk == 1:
                action = SGD_STEPS[min(step_idx, len(SGD_STEPS)-1)]
                turns.append({"speaker": "agent", "text": utt, "action": action, "step_index": min(step_idx, len(SGD_STEPS)-1)})
                step_idx += 1
            else:
                turns.append({"speaker": "user", "text": utt, "action": "none", "step_index": -1})

        variant = "common"
        dialogues.append({
            "turns": turns, "sop_id": "sgd_service",
            "variant": variant, "sop_steps": SGD_STEPS,
            "mandatory_before": {"confirm_action": ["collect_parameters", "call_api"]},
        })

    for d in dialogues[int(len(dialogues)*0.8):]:
        d["variant"] = "rare"

    with open(cache_path, "w") as f:
        json.dump(dialogues, f, indent=1)
    print(f"[SGD] Loaded {len(dialogues)} dialogues.")
    return dialogues[:n]


# ============================================================
# DSTC2 data loading (restaurant information domain)
# ============================================================
# Aligned with the manual annotations in dstc2_processed.json
# (greet_customer=0, collect_info=1, process_request=2,
#  recommend_restaurant=3, provide_details=4).
DSTC2_STEPS = [
    "greet_customer",
    "collect_info",           # ask for food type / area / price range
    "process_request",        # query the DB with collected constraints
    "recommend_restaurant",   # offer a matching restaurant
    "provide_details",        # phone / address / postcode on request
]

DSTC2_MANDATORY = {
    "recommend_restaurant": ["collect_info"],
    "provide_details": ["recommend_restaurant"],
}


def load_dstc2(n: int = 150) -> List[dict]:
    """Load DSTC2 dialogues from data/dstc2_processed.json.

    Accepts either the unified format used by multiwoz/sgd caches
    (turns with speaker/text/action/step_index) or a minimal format with
    just speaker+text turns — missing fields are normalized here.
    Returns [] (with a notice) if the data file is not present yet.
    """
    cache_path = os.path.join(DATA_DIR, "dstc2_processed.json")
    if not os.path.exists(cache_path):
        print(f"[DSTC2] {cache_path} not found — skipping dstc2 "
              f"(drop the processed data there to enable it).")
        return []

    with open(cache_path) as f:
        raw = json.load(f)

    dialogues = []
    for item in raw:
        turns_in = item.get("turns", [])
        if len(turns_in) < 4:
            continue

        turns = []
        step_idx_seq = 0
        for t in turns_in:
            speaker = t.get("speaker", "")
            speaker = "agent" if speaker in ("agent", "system", "sys") else "user"
            text = t.get("text", t.get("utterance", "")).strip()
            if not text:
                continue
            if speaker == "agent":
                action = t.get("action")
                if action in DSTC2_STEPS:
                    step_index = DSTC2_STEPS.index(action)
                else:
                    # No (or unknown) annotation: map sequentially like the
                    # multiwoz/sgd loaders do.
                    step_index = min(step_idx_seq, len(DSTC2_STEPS) - 1)
                    action = DSTC2_STEPS[step_index]
                turns.append({"speaker": "agent", "text": text,
                              "action": action, "step_index": step_index})
                step_idx_seq += 1
            else:
                turns.append({"speaker": "user", "text": text,
                              "action": "none", "step_index": -1})

        if not turns:
            continue

        dialogues.append({
            "turns": turns,
            "sop_id": item.get("sop_id", "dstc2_restaurant"),
            "variant": item.get("variant", "common"),
            "sop_steps": item.get("sop_steps", DSTC2_STEPS),
            "mandatory_before": item.get("mandatory_before", DSTC2_MANDATORY),
        })

    # If the source data carries no variant labels, mark the tail ~20% as rare
    # for consistency with the other datasets.
    if dialogues and all(d["variant"] == "common" for d in dialogues):
        for d in dialogues[int(len(dialogues) * 0.8):]:
            d["variant"] = "rare"

    # Rare dialogues sit at the tail of the source file; interleave 4:1 so any
    # prefix slice (e.g. the 40 used for the main table) keeps the ~20% mix.
    common = [d for d in dialogues if d["variant"] == "common"]
    rare = [d for d in dialogues if d["variant"] != "common"]
    mixed, ri = [], 0
    for i, d in enumerate(common):
        mixed.append(d)
        if (i + 1) % 4 == 0 and ri < len(rare):
            mixed.append(rare[ri])
            ri += 1
    mixed.extend(rare[ri:])
    dialogues = mixed

    print(f"[DSTC2] Loaded {len(dialogues)} dialogues.")
    return dialogues[:n]


def _generate_synthetic_tod(dataset_name: str, n: int) -> List[dict]:
    """Generate synthetic task-oriented dialogues as fallback."""
    print(f"[{dataset_name}] Generating {n} synthetic dialogues...")

    if dataset_name == "multiwoz":
        steps = MULTIWOZ_BOOKING_STEPS
        domains = ["hotel", "restaurant", "train", "attraction", "taxi"]
        mandatory = {"confirm_booking": ["collect_slots", "search_options"]}
    else:
        steps = ["greet", "identify_service", "collect_parameters", "call_api",
                 "present_results", "collect_more_info", "confirm_action", "provide_details", "close"]
        domains = ["restaurant_reservation", "flight_booking", "hotel_search", "weather", "bank"]
        mandatory = {"confirm_action": ["collect_parameters", "call_api"]}

    dialogues = []
    for i in range(n):
        domain = random.choice(domains)
        variant = "common" if i < int(n * 0.8) else "rare"
        sop_data = {"id": f"{dataset_name}_{domain}", "name": f"{domain} service",
                     "steps": steps, "branches": {}, "mandatory_before": mandatory}
        dlg = _generate_dialogue(sop_data, variant)
        dlg["sop_id"] = f"{dataset_name}_{domain}"
        dlg["variant"] = variant
        dlg["sop_steps"] = steps
        dlg["mandatory_before"] = mandatory
        dialogues.append(dlg)

    cache_path = os.path.join(DATA_DIR, f"{dataset_name}_processed.json")
    with open(cache_path, "w") as f:
        json.dump(dialogues, f, indent=1)
    return dialogues
