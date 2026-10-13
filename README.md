# Institutional Communication and Inequality: Code

An anonymized, compact code release for the paper. It includes the main data preparation, textual-legibility, institutional-response, neighborhood, homogenization, requester-assistance, and validation analyses. Data and generated model outputs are not distributed here.

AI-authorship/likeness detection code is intentionally excluded.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/scripts"
```

`vllm` is needed only for local LLM generation. `openai` is needed only for the optional remote batch runner. The statistical analyses run without either backend.

## Data

The analyses use public CFPB complaints, NHTSA complaints, San Diego service requests, federal FOIA logs, and ACS ZCTA attributes. Download helpers are provided:

```bash
python scripts/prepare_cfpb_2021_2024.py --out data/cfpb_2021_2024.csv
python scripts/download_narrative_response_datasets.py nhtsa --outdir data/raw/nhtsa --combined data/nhtsa_2021_2024.csv
python scripts/download_narrative_response_datasets.py sandiego311 --outdir data/raw/sandiego --combined data/sandiego_2021_2024.csv
python scripts/download_narrative_response_datasets.py foia --outdir data/raw/foia --combined data/foia_2021_2024.csv
python scripts/download_acs_zcta.py --year 2023 --out data/acs_zcta_2023.csv
```

Source portals can change. The download scripts document the URLs and expected source columns. No private or restricted data are required.

## Core Analysis

Build fixed-scale textual-legibility panels, then estimate temporal changes, response returns, neighborhood gaps, and institutional heterogeneity:

```bash
python scripts/build_corrected_narrative_panel.py cfpb --input data/cfpb_2021_2024.csv --acs data/acs_zcta_2023.csv --outdir outputs/cfpb/panel
python scripts/run_corrected_actionability_models.py cfpb --paneldir outputs/cfpb/panel --acs data/acs_zcta_2023.csv --outdir outputs/cfpb/models
```

Repeat both commands with `nhtsa`, `sandiego`, and `foia`. Internally, legacy files call the textual-legibility score `actionability`; this is the continuous five-component legibility measure reported in the paper.

Cross-domain summaries and robustness analyses:

```bash
python scripts/run_unified_legibility_summaries.py --root outputs --outdir outputs/summary
python scripts/run_outcome_sensitivity_grid.py cfpb --paneldir outputs/cfpb/panel --outdir outputs/cfpb/sensitivity
python scripts/run_time_to_response_models.py sandiego --paneldir outputs/sandiego/panel --outdir outputs/sandiego/survival
```

## Additional NLP Analyses

The following scripts implement the paper's topic/case-mix decomposition, rhetorical-move analysis, lexical and semantic homogenization, and information-versus-polish analysis:

```text
run_topic_case_mix_analysis.py
run_rhetorical_move_analysis.py
run_voice_homogenization_analysis.py
run_semantic_voice_convergence.py
run_information_polish_analysis.py
run_institution_conversion_profiles.py
```

Each script exposes its inputs and options through `python scripts/<name>.py --help`.

## Requester-Side Audits

Prepare prompts with the three audit scripts, generate outputs with either `run_research_prompts_vllm.py` or `run_openai_grouped_batch.py`, and invoke the corresponding `analyze` subcommand:

```text
run_counterfactual_register_audit.py       # constrained rewriting
run_communication_benchmark.py             # missing-information elicitation
run_requester_assistance_simulation.py      # assistance-policy comparison
run_neighborhood_assistance_audit.py        # neighborhood contrasts
```

API credentials must be supplied through environment variables and are never stored in this repository.

## Reproducibility Notes

- The public-diffusion cutoff is November 30, 2022.
- Scores are standardized against each dataset's pre-cutoff distribution and are comparable over time within a domain, not as absolute levels across domains.
- All reported associations are observational.
- Randomized sampling scripts use fixed seeds; large-data scripts use checkpointed or streaming reads.
