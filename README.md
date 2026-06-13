# Stateguard-code
A repository for stateguard code
# StateGuard: Auditable Workflow-State Monitoring for SOP-Compliant LLM Agents

Anonymous code and data release for the StateGuard paper (EMNLP Industry Track
submission). StateGuard is a lightweight runtime monitor that enforces a
Standard Operating Procedure (SOP) on a black-box LLM agent: it distills a
probabilistic workflow-state model from action logs, blocks and regenerates
prerequisite-violating actions with a hard screen, and issues belief-entropy
graded guidance. The per-turn belief trajectory doubles as an audit log.

> This repository is anonymized for double-blind review. It contains no author
> or institution identifiers. API endpoints and keys are read from environment
> variables; no credentials are stored in the code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The code talks to an OpenAI-compatible endpoint. Set your own credentials:
export STATEGUARD_API_KEY=sk-...
export STATEGUARD_API_BASE=https://api.openai.com/v1   # or any OpenAI-style proxy
# Optional secondary provider for models not served by API_BASE:
# export STATEGUARD_MODELHUB_URL=...
# export STATEGUARD_MODELHUB_AK=...
```

Model names used as agents/judges are configured in `config.py`.

## Repository layout

```
config.py               # models, dataset sizes, PFSA/guidance hyperparameters
llm_utils.py            # OpenAI-style chat client (retry + param adaptation)
data_loader.py          # dataset loading and SOP mapping
run_all.py              # main evaluation (StateGuard + baselines, 4 datasets)
run_ablation.py         # component / automaton-precision ablations
run_robustness.py       # common / rare / adversarial robustness subsets
run_model_agnostic.py   # cross-backend generalization
run_adaptation.py       # adaptation from N example dialogues
run_fsm_baseline.py     # hand-authored FSM controller baseline
make_belief_ablation.py # offline belief-vs-uniform diagnostic (no API calls)
make_anomaly_analysis.py# AUROC of PFSA predictive mass for violation detection
make_audit_sample.py    # build the human action-extraction audit worksheet
make_audit_score.py     # score the completed human audit
results/                # checkpoints, aggregated metrics, judge outputs
data/                   # ProcBench SOPs and dialogues; mapped public datasets
```

## Reproducing the main results

```bash
python run_all.py            # main table (PCR / SVR / TSR / Fluency)
python run_robustness.py     # robustness subsets
python run_ablation.py       # ablations
python run_model_agnostic.py # cross-backend table
python run_adaptation.py     # adaptation curve
```

Aggregated metrics are written under `results/`. Generation and judging both
issue LLM calls, so a funded endpoint is required to re-run from scratch;
precomputed checkpoints are included under `results/checkpoints/`.

## Human validation

The action classifier that underlies all metrics was audited against a human
annotator on 180 ProcBench turns (150 common, 15 rare, 15 adversarial).

```bash
python make_audit_sample.py        # regenerate the common worksheet
python make_audit_sample_rare.py   # regenerate the rare/adversarial worksheet
python make_audit_score.py         # score audit_sample_tag.csv + *_rare_tag.csv
```

The completed annotations (`audit_sample_tag.csv`, `audit_sample_rare_tag.csv`)
and the scoring output (`results/audit_score.json`) are included. Overall
exact-match accuracy is 86.7% (91.1% on StateGuard turns).

## ProcBench

`data/` contains ProcBench: 675 dialogues over 15 finance, healthcare, and
telecom SOPs, each with common, rare, and adversarial variants. SOPs are JSON
schemas with step sequences, conditional branches, and mandatory prerequisites.
