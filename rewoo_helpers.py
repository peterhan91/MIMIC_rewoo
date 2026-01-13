"""Helper utilities for the ReWOO agent.

Includes lab name -> itemid mapping and imaging input parsing.
"""

from __future__ import annotations

from typing import Dict, List, Union
import re

from thefuzz import process as fuzz_process


def convert_labs_to_itemid(
    tests: List[str],
    lab_test_mapping_df,
) -> List[Union[int, str]]:
    """Map lab names to itemids using fuzzy label matching.

    - If a label fuzzy-matches (ratio >=90), expands to its corresponding_ids list.
    - Else returns the original string (the downstream tool will report N/A).
    """
    labels = lab_test_mapping_df["label"].tolist() if "label" in lab_test_mapping_df.columns else []
    out: List[Union[int, str]] = []
    for raw in tests:
        name = str(raw).strip()
        if not name:
            continue
        match_label = None
        if labels:
            m, score = fuzz_process.extractOne(name, labels)
            if score >= 90:
                match_label = m
        if match_label is None:
            out.append(name)
            continue
        # Expand to itemids
        try:
            cids = lab_test_mapping_df.loc[
                lab_test_mapping_df["label"] == match_label, "corresponding_ids"
            ].iloc[0]
            if isinstance(cids, list) and cids:
                out.extend(cids)
            else:
                out.append(name)
        except Exception:
            out.append(name)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Union[int, str]] = []
    for x in out:
        key = ("id", int(x)) if isinstance(x, int) else ("name", str(x))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)
    return uniq


REGION_KEYWORDS = {
    "abdomen": "Abdomen",
    "abdominal": "Abdomen",
    "rlq": "Abdomen",
    "llq": "Abdomen",
    "ruq": "Abdomen",
    "luq": "Abdomen",
    "gallbladder": "Abdomen",
    "appendix": "Abdomen",
    "chest": "Chest",
    "cxr": "Chest",
    "lung": "Chest",
    "head": "Head",
    "brain": "Head",
    "neck": "Neck",
    "carotid": "Neck",
    "pelvis": "Pelvis",
}

MODALITY_KEYWORDS = {
    "ultrasound": "Ultrasound",
    "sonography": "Ultrasound",
    "us": "Ultrasound",
    "ct": "CT",
    "mri": "MRI",
    "mr": "MRI",
    "radiograph": "Radiograph",
    "x-ray": "Radiograph",
    "xray": "Radiograph",
    "cxr": "Radiograph",
    # Unique modalities (auto-mapped downstream to broad modality)
    "ctu": "CTU",
    "mrcp": "MRCP",
    "mra": "MRA",
    "mre": "MRE",
    "ercp": "ERCP",
}


def parse_imaging_action_input(text: str) -> Dict[str, str]:
    """Extract modality and region from free text using simple keyword maps."""
    s = str(text or "").lower()
    reg = None
    mod = None
    for k, v in REGION_KEYWORDS.items():
        if re.search(rf"\b{k}\b", s):
            reg = v
            break
    for k, v in MODALITY_KEYWORDS.items():
        if re.search(rf"\b{k}\b", s):
            mod = v
            break
    if reg is None and "carotid" in s:
        reg = "Neck"
    if not reg or not mod:
        raise ValueError("Imaging input must contain a region and modality")
    return {"region": reg, "modality": mod}
