"""Scan checkpoints for corruption left by the balance outage (empty agent
text, all-'none' classifications, fallback 3.5 fluency) and optionally clean.

Usage:
  .venv/bin/python scan_checkpoints.py          # report only
  .venv/bin/python scan_checkpoints.py --clean  # truncate/delete corrupt parts
"""
import argparse
import glob
import json
import os

CKPT = "results/checkpoints"


def dialogue_corrupt_generation(d):
    """Generation corrupt: any agent turn with empty text."""
    return any(t.get("speaker") == "agent" and not (t.get("text") or "").strip()
               for t in d.get("turns", []))


def dialogue_corrupt_classification(d):
    """Classification corrupt: ALL agent turns classified 'none' despite text."""
    agent = [t for t in d.get("turns", []) if t.get("speaker") == "agent"
             and (t.get("text") or "").strip()]
    if not agent:
        return False
    return all(t.get("action", "none") == "none" for t in agent)


def scan_jsonl(path, check):
    """Return (n_total, first_bad_index or None)."""
    first_bad = None
    n = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            n += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                if first_bad is None:
                    first_bad = i
                break
            if first_bad is None and check(d):
                first_bad = i
    return n, first_bad


def truncate_jsonl(path, keep):
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    with open(path, "w") as f:
        f.writelines(lines[:keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    issues = 0

    for path in sorted(glob.glob(os.path.join(CKPT, "main_*.jsonl"))):
        n, bad = scan_jsonl(path, dialogue_corrupt_generation)
        if bad is not None:
            issues += 1
            print(f"[GEN ] {os.path.basename(path)}: {n} dialogues, "
                  f"first corrupt at #{bad}" + (" -> truncated" if args.clean else ""))
            if args.clean:
                truncate_jsonl(path, bad)
        else:
            print(f"[ ok ] {os.path.basename(path)}: {n} dialogues")

    for path in sorted(glob.glob(os.path.join(CKPT, "jm_*.jsonl"))):
        n, bad = scan_jsonl(path, dialogue_corrupt_classification)
        if bad is not None:
            issues += 1
            print(f"[CLS ] {os.path.basename(path)}: {n} dialogues, "
                  f"first all-none at #{bad}" + (" -> truncated" if args.clean else ""))
            if args.clean:
                truncate_jsonl(path, bad)

    for path in sorted(glob.glob(os.path.join(CKPT, "flu_*.json"))):
        try:
            with open(path) as f:
                scores = json.load(f)
        except json.JSONDecodeError:
            scores = None
        frac_fallback = (sum(1 for s in (scores or []) if s == 3.5)
                         / max(len(scores or []), 1))
        if scores is None or frac_fallback > 0.2:
            issues += 1
            print(f"[FLU ] {os.path.basename(path)}: "
                  f"{'unreadable' if scores is None else f'{frac_fallback:.0%} fallback 3.5'}"
                  + (" -> deleted" if args.clean else ""))
            if args.clean:
                os.remove(path)

    print(f"\n{'CLEANED' if args.clean else 'FOUND'} {issues} corrupt checkpoint(s)"
          if issues else "\nAll checkpoints healthy")


if __name__ == "__main__":
    main()
