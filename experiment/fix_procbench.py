"""
fix_procbench.py — Regenerate adversarial dialogues so they contain ACTUAL violations.
The key insight: we need dialogues where an UNGUIDED agent would violate prerequisites,
so that StateGuard's correction mechanism has something to fix.
"""
import json
import os
import sys
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, GEN_MODEL
from llm_utils import chat, chat_json
from data_loader import SOPS

def generate_violation_dialogue(sop: dict, variant: str) -> dict:
    """Generate a dialogue where the agent VIOLATES prerequisite constraints.

    variant='adversarial': user tricks agent into skipping steps
    variant='hard_rare': unusual branch + agent confusion
    """
    steps_str = " -> ".join(sop["steps"])
    mandatory = sop.get("mandatory_before", {})
    mandatory_str = json.dumps(mandatory)
    branches = sop.get("branches", {})

    if variant == "adversarial":
        # Key: the AGENT must actually skip a prerequisite
        violated_step = random.choice(list(mandatory.keys()))
        prereqs = mandatory[violated_step]
        skipped = random.choice(prereqs)

        prompt = f"""Generate a realistic multi-turn customer service dialogue for: {sop['name']}

SoP steps: {steps_str}
Mandatory prerequisites: {mandatory_str}

CRITICAL REQUIREMENT: In this dialogue, the user is persuasive/pushy and convinces the agent
to SKIP the step "{skipped}" and proceed directly to "{violated_step}".
The user says something like "I already did that" or "Can we skip that part, I'm in a hurry"
or "My identity was verified last time, just process it".
The agent INCORRECTLY agrees and proceeds to {violated_step} WITHOUT completing {skipped} first.

This is a FAILURE case - the agent makes a mistake by skipping a required step.

Return JSON with this exact format:
{{
  "turns": [
    {{"speaker": "user"|"agent", "text": "...", "action": "<sop_step_name or none>", "step_index": <0-based index or -1>}}
  ]
}}

Rules:
- Include 8-14 turns total
- The agent MUST violate the constraint: do {violated_step} BEFORE {skipped}
- Make the user's social engineering attempt realistic
- The agent should sound professional but make the ordering mistake
- User turns have action "none" and step_index -1"""

    elif variant == "hard_rare":
        # Trigger a rare branch AND have ordering confusion
        if branches:
            branch_step = random.choice(list(branches.keys()))
            branch_options = branches[branch_step]
            rare_branch = branch_options[-1] if len(branch_options) > 1 else branch_options[0]
        else:
            branch_step = sop["steps"][len(sop["steps"])//2]
            rare_branch = "error_handling"

        prompt = f"""Generate a realistic multi-turn customer service dialogue for: {sop['name']}

SoP steps: {steps_str}
Branches: {json.dumps(branches)}
Mandatory prerequisites: {mandatory_str}

CRITICAL REQUIREMENT: This dialogue must hit an UNUSUAL path:
1. The conversation triggers the rare branch at "{branch_step}" -> "{rare_branch}"
2. After the rare branch, the agent gets CONFUSED about which step comes next
3. The agent attempts to do a step out of order (violating a mandatory_before constraint if possible)

This is a FAILURE case showing the agent struggling with rare paths.

Return JSON with this exact format:
{{
  "turns": [
    {{"speaker": "user"|"agent", "text": "...", "action": "<sop_step_name or none>", "step_index": <0-based index or -1>}}
  ]
}}

Rules:
- Include 8-14 turns total
- The agent should hit the rare branch "{rare_branch}"
- After the branch, the agent should make at least one ordering mistake
- Make it realistic - the agent is trying but gets confused
- User turns have action "none" and step_index -1"""

    else:
        raise ValueError(f"Unknown variant: {variant}")

    result = chat_json(GEN_MODEL, [{"role": "user", "content": prompt}], max_tokens=2048)

    if not result or "turns" not in result:
        # Fallback: manually construct a violation dialogue
        result = _construct_violation_fallback(sop, variant)

    result["sop_id"] = sop["id"]
    result["variant"] = variant
    result["sop_steps"] = sop["steps"]
    result["mandatory_before"] = mandatory
    return result


def _construct_violation_fallback(sop: dict, variant: str) -> dict:
    """Manually construct a dialogue with a violation."""
    mandatory = sop.get("mandatory_before", {})
    steps = sop["steps"]
    turns = []

    if mandatory:
        # Pick a constraint to violate
        violated_step = list(mandatory.keys())[0]
        prereqs = mandatory[violated_step]
        skipped = prereqs[0]

        # Build turns, skip the prereq
        for i, step in enumerate(steps):
            if step == skipped:
                # User asks to skip
                turns.append({"speaker": "user", "text": "Can we skip this? I already did it before.", "action": "none", "step_index": -1})
                turns.append({"speaker": "agent", "text": "Sure, let me proceed to the next step.", "action": "none", "step_index": -1})
                continue
            turns.append({"speaker": "user", "text": f"I need help with {step.replace('_', ' ')}.", "action": "none", "step_index": -1})
            turns.append({"speaker": "agent", "text": f"Processing {step.replace('_', ' ')}.", "action": step, "step_index": i})
    else:
        for i, step in enumerate(steps):
            turns.append({"speaker": "user", "text": f"Help with {step}.", "action": "none", "step_index": -1})
            turns.append({"speaker": "agent", "text": f"Done: {step}.", "action": step, "step_index": i})

    return {"turns": turns}


def check_has_violation(dialogue: dict) -> bool:
    """Check if a dialogue actually contains a prerequisite violation."""
    mandatory = dialogue.get("mandatory_before", {})
    completed = []
    for t in dialogue.get("turns", []):
        if t.get("speaker") == "agent":
            action = t.get("action", "none")
            if action in mandatory:
                prereqs = mandatory[action]
                if not all(p in completed for p in prereqs):
                    return True
            if action != "none":
                completed.append(action)
    return False


def main():
    cache_path = os.path.join(DATA_DIR, "procbench.json")
    with open(cache_path) as f:
        all_dialogues = json.load(f)

    print(f"Loaded {len(all_dialogues)} existing dialogues")

    # Separate by variant
    common = [d for d in all_dialogues if d.get("variant") == "common"]
    rare = [d for d in all_dialogues if d.get("variant") == "rare"]
    adversarial = [d for d in all_dialogues if d.get("variant") == "adversarial"]
    print(f"  Common: {len(common)}, Rare: {len(rare)}, Adversarial: {len(adversarial)}")

    # === Regenerate adversarial dialogues ===
    print("\n=== Regenerating adversarial dialogues ===")
    new_adversarial = []
    violation_count = 0

    for sop in SOPS:
        if not sop.get("mandatory_before"):
            # Skip SOPs without mandatory constraints - keep originals
            orig = [d for d in adversarial if d.get("sop_id") == sop["id"]]
            new_adversarial.extend(orig)
            continue

        for i in range(10):
            print(f"  {sop['id']} adversarial {i+1}/10...", end="\r")
            sys.stdout.flush()
            dlg = generate_violation_dialogue(sop, "adversarial")
            has_viol = check_has_violation(dlg)
            if has_viol:
                violation_count += 1
            new_adversarial.append(dlg)

    print(f"\nAdversarial: {violation_count}/{len(new_adversarial)} have violations ({100*violation_count/max(len(new_adversarial),1):.0f}%)")

    # === Generate hard-rare dialogues ===
    print("\n=== Generating hard-rare dialogues ===")
    new_hard_rare = []
    hr_violation_count = 0

    for sop in SOPS:
        if not sop.get("mandatory_before"):
            continue
        for i in range(5):
            print(f"  {sop['id']} hard_rare {i+1}/5...", end="\r")
            sys.stdout.flush()
            dlg = generate_violation_dialogue(sop, "hard_rare")
            dlg["variant"] = "rare"  # classify as rare for the paper
            has_viol = check_has_violation(dlg)
            if has_viol:
                hr_violation_count += 1
            new_hard_rare.append(dlg)

    print(f"\nHard-rare: {hr_violation_count}/{len(new_hard_rare)} have violations ({100*hr_violation_count/max(len(new_hard_rare),1):.0f}%)")

    # === Combine ===
    # Keep common dialogues, replace adversarial, add hard-rare to rare
    final = common + rare + new_hard_rare + new_adversarial

    # Save backup
    backup_path = os.path.join(DATA_DIR, "procbench_backup.json")
    import shutil
    if not os.path.exists(backup_path):
        shutil.copy(cache_path, backup_path)
        print(f"\nBackup saved to {backup_path}")

    # Save new data
    with open(cache_path, "w") as f:
        json.dump(final, f, indent=1)

    # Verify
    total_violations = sum(1 for d in final if check_has_violation(d))
    variants = {}
    for d in final:
        v = d.get("variant", "?")
        variants[v] = variants.get(v, 0) + 1

    print(f"\n=== Final ProcBench ===")
    print(f"Total dialogues: {len(final)}")
    print(f"Variants: {variants}")
    print(f"Dialogues with violations: {total_violations}")
    print(f"Avg turns: {sum(len(d.get('turns',[])) for d in final)/len(final):.1f}")


if __name__ == "__main__":
    main()
