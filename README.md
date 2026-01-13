# MIMIC_rewoo (ReWOO-only runner)

This folder contains the minimal files needed to run the ReWOO agent for diagnosis using MIMIC-III/IV HADM pickles.
Hugging Face chat models with `apply_chat_template()` support are supported (e.g., Llama, Qwen3, LFM2.5).

Included:
- `run.py` (ReWOO-only)
- `rewoo_agent.py`, `rewoo_helpers.py`, `tools.py`
- Lab mappings and reference ranges
- Sample HADM pickles under `data/`

Behavior:
- Tools limited to Physical Examination, Laboratory Tests, Imaging, ECG, and Echocardiogram.
- Output is a single final diagnosis string (no treatments or extra text).
- Writes per-run outputs and a structured JSONL event log for debugging.

## Quickstart

MIMIC-IV example:

```bash
python run.py \
  --hadm-pkl data/CDM_IV/appendicitis_hadm_info_first_diag.pkl \
  --lab-map-pkl lab_test_mapping_IV.pkl \
  --ref-ranges-json itemid_ref_ranges_IV.json \
  --hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct
```

MIMIC-III example:

```bash
python run.py \
  --hadm-pkl data/CDM_III/myocardial_infarction_hadm_info_first_diag.pkl \
  --lab-map-pkl lab_test_mapping_III.pkl \
  --ref-ranges-json itemid_ref_ranges_III.json \
  --hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct
```

Debug model (Qwen3; requires `transformers>=4.51.0`):

```bash
python run.py \
  --hadm-pkl data/CDM_IV/appendicitis_hadm_info_first_diag.pkl \
  --lab-map-pkl lab_test_mapping_IV.pkl \
  --ref-ranges-json itemid_ref_ranges_IV.json \
  --hf-model-id Qwen/Qwen3-4B-Instruct-2507
```

## Dependencies

You will need (at least):
- `torch`
- `transformers`
- `huggingface_hub`
- `pandas`
- `thefuzz`
- `tqdm`

Install your preferred versions according to your environment.

## Outputs

Each run writes to `outputs_plain/<run_name>/` by default:
- `<run_name>_results.json` (final predictions)
- `events.jsonl` (structured planner/worker/solver logs)

Disable structured logging with `--disable-event-log` or rename it with `--event-log-name`.

## Tests

```bash
python -m unittest tests.rewoo_agent_test
```
