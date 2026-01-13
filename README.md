# MIMIC_rewoo (ReWOO-only runner)

This folder contains the minimal files needed to run the ReWOO agent for diagnosis using MIMIC-III/IV HADM pickles.
Only Llama-based Hugging Face chat models are supported.

Included:
- `run.py` (ReWOO-only)
- `rewoo_agent.py`, `rewoo_helpers.py`, `tools.py`
- Lab mappings and reference ranges
- Sample HADM pickles under `data/`

Behavior:
- Tools limited to Physical Examination, Laboratory Tests, Imaging, ECG, and Echocardiogram.
- Output is a single final diagnosis string (no treatments or extra text).

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

## Dependencies

You will need (at least):
- `torch`
- `transformers`
- `huggingface_hub`
- `pandas`
- `thefuzz`
- `tqdm`

Install your preferred versions according to your environment.
