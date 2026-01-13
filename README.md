# MIMIC_simple (ReWOO-only runner)

This folder contains the minimal files needed to run the ReWOO agent for diagnosis using MIMIC-III/IV HADM pickles.

Included:
- `run.py` (use `--agent-type rewoo`)
- `rewoo_agent.py`, `react_agent.py`, `tools.py`
- Lab mappings and reference ranges
- Sample HADM pickles under `data/`
- `icd_procedures_index.json` for procedure recommendation lookup

## Quickstart

MIMIC-IV example:

```bash
python run.py \
  --agent-type rewoo \
  --hadm-pkl data/CDM_IV/appendicitis_hadm_info_first_diag.pkl \
  --lab-map-pkl lab_test_mapping_IV.pkl \
  --ref-ranges-json itemid_ref_ranges_IV.json \
  --hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct
```

MIMIC-III example:

```bash
python run.py \
  --agent-type rewoo \
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
