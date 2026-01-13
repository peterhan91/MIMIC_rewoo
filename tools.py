"""Pure Python implementations of project tools (no internal imports).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
import re


# Mapping for unique modalities to their broad modality equivalents
UNIQUE_TO_BROAD_MODALITY: Dict[str, str] = {
    "CTU": "CT",
    "Carotid ultrasound": "Ultrasound",
    "EUS": "Ultrasound",
    "MRCP": "MRI",
    "ERCP": "Radiograph",
    "Upper GI Series": "Radiograph",
    "MRA": "MRI",
    "MRE": "MRI",
}


def _itemid_to_field(itemid: int, field: str, lab_test_mapping_df) -> Any:
    """Return field for given itemid from mapping DataFrame.

    Expects columns: 'itemid' and the requested field (e.g., 'label', 'fluid').
    """
    row = lab_test_mapping_df.loc[lab_test_mapping_df["itemid"] == itemid, field]
    return row.iloc[0]


def _lookup_numeric_dict(container: Dict[Any, Any], key: int) -> Tuple[bool, Any]:
    if not isinstance(container, dict):
        return False, None
    if key in container:
        return True, container[key]
    key_str = str(key)
    if key_str in container:
        return True, container[key_str]
    return False, None


def physical_examination(action_results: Dict[str, Any]) -> str:
    """Return physical examination observations with header.

    action_results must contain key 'Physical Examination' or it will report 'Not available.'.
    """
    obs = action_results.get("Physical Examination", "Not available.")
    return f"Physical Examination:\n{obs}\n"


def _create_lab_test_string(
    test_id: Union[int, str],
    lab_test_mapping_df,
    hadm_info: Dict[str, Any],
    include_ref_range: bool = False,
    bin_lab_results: bool = False,
) -> str:
    # Label, value, and fluid
    lab_test_fluid = _itemid_to_field(test_id, "fluid", lab_test_mapping_df)
    lab_test_label = _itemid_to_field(test_id, "label", lab_test_mapping_df)
    labs = hadm_info.get("Laboratory Tests", {}) or {}
    micro = hadm_info.get("Microbiology", {}) or {}
    ref_low = hadm_info.get("Reference Range Lower", {}) or {}
    ref_up = hadm_info.get("Reference Range Upper", {}) or {}

    lab_found, lab_test_value = _lookup_numeric_dict(labs, test_id)
    if not lab_found:
        lab_test_value = "N/A"

    # If test not found in lab tests, check microbiology
    if lab_test_value == "N/A":
        lab_test_fluid = "Microbiology"
        micro_found, micro_val = _lookup_numeric_dict(micro, test_id)
        lab_test_value = micro_val if micro_found else "N/A"

    # Optional binning to Low/Normal/High
    if bin_lab_results:
        if include_ref_range:
            raise ValueError("Binning and printing ref range concurrently not supported")
        _, rr_lower = _lookup_numeric_dict(ref_low, test_id)
        _, rr_upper = _lookup_numeric_dict(ref_up, test_id)
        if rr_lower == rr_lower and rr_upper == rr_upper:  # ensure not NaN
            try:
                v = float(str(lab_test_value).split()[0])
                if v < rr_lower:
                    lab_test_value = "Low"
                elif v > rr_upper:
                    lab_test_value = "High"
                else:
                    lab_test_value = "Normal"
            except Exception:
                pass

    # Base string
    s = f"({lab_test_fluid}) {lab_test_label}: {lab_test_value}"

    # Optional reference range annotation
    if include_ref_range:
        rr_lower = hadm_info.get("Reference Range Lower", {}).get(test_id, None)
        rr_upper = hadm_info.get("Reference Range Upper", {}).get(test_id, None)
        if rr_lower == rr_lower and rr_upper == rr_upper:  # not NaN
            s += f" | RR: [{rr_lower} - {rr_upper}]"

    return s + "\n"


def _format_lab_with_calculator(
    test_id: Union[int, str],
    lab_test_mapping_df,
    hadm_info: Dict[str, Any],
    ref_ranges: Optional[Dict[str, Any]] = None,
    critical_pct: float = 30.0,
    include_units: bool = True,
    strict_compat: bool = False,
) -> str:
    """Calculator-style lab formatting with severity and % deviation.

    Enhanced to cover fluids and microbiology:
    - Uses actual fluid from mapping for prefix; defaults to 'Blood' if missing.
    - Falls back to Microbiology when lab value is N/A; returns qualitative line.
    """
    label = _itemid_to_field(test_id, "label", lab_test_mapping_df)
    try:
        mapping_fluid = _itemid_to_field(test_id, "fluid", lab_test_mapping_df)
    except Exception:
        mapping_fluid = None

    # Retrieve value from labs first; optionally fall back to microbiology (non-strict mode)
    labs = hadm_info.get("Laboratory Tests", {}) or {}
    micro = hadm_info.get("Microbiology", {}) or {}
    ref_low = hadm_info.get("Reference Range Lower", {}) or {}
    ref_up = hadm_info.get("Reference Range Upper", {}) or {}

    found_lab, value_str = _lookup_numeric_dict(labs, test_id)
    if not found_lab:
        value_str = "N/A"
    value_source = "labs"
    if (not strict_compat) and value_str == "N/A":
        micro_val = hadm_info.get("Microbiology", {}).get(test_id, "N/A")
        if micro_val != "N/A":
            value_str = micro_val
            value_source = "micro"

    # If non-numeric or not present, return qualitative with appropriate fluid
    toks = str(value_str).split()
    try:
        val = float(toks[0])
    except Exception:
        if strict_compat:
            return f"(Blood) {label}: {value_str}\n"
        # Choose fluid label
        if value_source == "micro":
            fluid_name = "Microbiology"
        else:
            fluid_name = mapping_fluid if mapping_fluid == mapping_fluid else None  # handle NaN
            fluid_name = fluid_name or "Blood"
        return f"({fluid_name}) {label}: {value_str}\n"

    unit = toks[1] if include_units and len(toks) > 1 else ""

    # Range lookup: prefer external ref_ranges, else fallback to hadm
    ref_ranges = ref_ranges or {}
    rr = ref_ranges.get(str(test_id)) or {}
    lower = rr.get("ref_range_lower")
    upper = rr.get("ref_range_upper")
    if (lower in (None, "N/A")) or (upper in (None, "N/A")):
        _, l2 = _lookup_numeric_dict(ref_low, test_id)
        _, u2 = _lookup_numeric_dict(ref_up, test_id)
        lower = l2 if lower in (None, "N/A") else lower
        upper = u2 if upper in (None, "N/A") else upper

    # If ranges are not numeric, return qualitative with fluid
    try:
        lower_f = float(lower)
        upper_f = float(upper)
    except Exception:
        if strict_compat:
            return f"(Blood) {label}: {value_str}\n"
        fluid_name = mapping_fluid if mapping_fluid == mapping_fluid else None
        fluid_name = fluid_name or "Blood"
        return f"({fluid_name}) {label}: {value_str}\n"

    # Classification
    status = "Normal"
    pct_str = ""
    if val < lower_f and lower_f > 0:
        diff = lower_f - val
        pct = max(0.0, (diff / lower_f) * 100.0)
        status = "LOW" if pct < critical_pct else "CRITICALLY LOW"
        pct_str = f", {pct:.1f}% below normal"
    elif val > upper_f and upper_f > 0:
        diff = val - upper_f
        pct = max(0.0, (diff / upper_f) * 100.0)
        status = "HIGH" if pct < critical_pct else "CRITICALLY HIGH"
        pct_str = f", {pct:.1f}% above normal"

    unit_str = f" {unit}" if unit else ""
    if strict_compat:
        # Match original formatting: no fluid prefix on numeric lines
        return (
            f"{label}: {status if status!='Normal' else 'Normal'} "
            f"({val:g}{unit_str}, normal: {lower_f:g}-{upper_f:g}{unit_str}{pct_str})\n"
        )
    else:
        # Enhanced: include fluid in numeric output too
        fluid_name = mapping_fluid if mapping_fluid == mapping_fluid else None
        fluid_name = fluid_name or "Blood"
        return (
            f"({fluid_name}) {label}: {status if status!='Normal' else 'Normal'} "
            f"({val:g}{unit_str}, normal: {lower_f:g}-{upper_f:g}{unit_str}{pct_str})\n"
        )


def lab_tests(
    action_results: Dict[str, Any],
    action_input: List[Union[int, str]],
    lab_test_mapping_df,
    include_ref_range: bool = False,
    bin_lab_results: bool = False,
    use_calculator: bool = False,
    ref_ranges: Optional[Dict[str, Any]] = None,
    calculator_critical_pct: float = 30.0,
    calculator_include_units: bool = True,
    strict_compat: bool = False,
) -> str:
    """Return lab results for requested tests (with header)."""
    result = []
    labs_dict = action_results.get("Laboratory Tests", {}) or {}
    micro_dict = action_results.get("Microbiology", {}) or {}
    for test in action_input:
        # Found numeric ID test in patient records
        if isinstance(test, int):
            has_lab, _ = _lookup_numeric_dict(labs_dict, test)
            has_micro, _ = _lookup_numeric_dict(micro_dict, test)
            if not has_lab and not has_micro:
                continue
            if use_calculator:
                result.append(
                    _format_lab_with_calculator(
                        test,
                        lab_test_mapping_df,
                        action_results,
                        ref_ranges=ref_ranges,
                        critical_pct=calculator_critical_pct,
                        include_units=calculator_include_units,
                        strict_compat=strict_compat,
                    )
                )
            else:
                result.append(
                    _create_lab_test_string(
                        test,
                        lab_test_mapping_df,
                        action_results,
                        include_ref_range=include_ref_range,
                        bin_lab_results=bin_lab_results,
                    )
                )
        # Requested but not matched / not present
        elif isinstance(test, str):
            result.append(f"{test}: N/A\n")
        # Else: silently ignore unmatched numeric IDs (as in original)

    body = "".join(result)
    return f"Laboratory Tests:\n{body}"


def imaging(
    action_results: Dict[str, Any],
    action_input: Dict[str, str],
    already_requested_scans: Optional[Dict[str, int]] = None,
) -> str:
    """Return imaging report for a region+modality (with header).

    Respects repeat-scan gating identical to the original tool.
    action_input must contain keys: 'region', 'modality'.
    """
    already_requested_scans = already_requested_scans or {}
    region = action_input.get("region")
    modality = action_input.get("modality")
    requested_scan = f"{region} {modality}"
    repeat_scan_index = already_requested_scans.get(requested_scan, 0)

    result: Optional[str] = None
    for rad in action_results.get("Radiology", []):
        mod = rad.get("Modality")
        reg = rad.get("Region")
        # Match if exact modality or if unique modality maps to requested broad modality
        modality_match = (mod == modality) or (UNIQUE_TO_BROAD_MODALITY.get(mod) == modality)
        if modality_match and reg == region:
            if repeat_scan_index == 0:
                report = rad.get("Report", "Not available. Try a different imaging modality.")
                if requested_scan not in already_requested_scans:
                    already_requested_scans[requested_scan] = 1
                else:
                    already_requested_scans[requested_scan] += 1
                return f"Imaging:\n{requested_scan}: {report}\n"
            else:
                result = "Cannot repeat this scan anymore. Try a different imaging modality."
                repeat_scan_index -= 1

    if not result:
        result = "Not available. Try a different imaging modality."
    return f"Imaging:\n{requested_scan}: {result}\n"


# ------------------------------- ECG (pure) -------------------------------

def ecg(action_results: Dict[str, Any]) -> str:
    """Return the earliest ECG report in a compact one-line format.

    - Expects action_results['ECG'] to be a list of dicts with keys
      like 'Report', 'Note ID', and 'Charttime'.
    - Selects the earliest entry by 'Charttime' when available; if no
      timestamps are available, falls back to the first list entry.
    - Output format: "ECG: <report>\n"
    """
    ecg_list = action_results.get("ECG", [])
    if not isinstance(ecg_list, list) or not ecg_list:
        return "ECG: Not available.\n"

    # Partition into entries with and without usable timestamps
    with_time = []
    without_time = []
    for idx, e in enumerate(ecg_list):
        ct = e.get("Charttime")
        has_time = (ct is not None) and (str(ct).lower() != "nat")
        (with_time if has_time else without_time).append((idx, e))

    if with_time:
        try:
            # Sort by Charttime ascending; use string form for deterministic ordering
            with_time.sort(key=lambda t: str(t[1].get("Charttime")))
        except Exception:
            # If sorting fails, preserve original order among with_time
            with_time.sort(key=lambda t: t[0])
        earliest_entry = with_time[0][1]
    else:
        # Fallback: first in provided order
        earliest_entry = without_time[0][1]

    report = str(earliest_entry.get("Report", "")).strip() or "Not available."
    return f"ECG: {report}\n"


# ------------------------------- Echo (pure) -------------------------------

def echocardiogram(action_results: Dict[str, Any]) -> str:
    """Return the earliest Echocardiogram (Echo) report in a compact format.

    - Expects action_results['Echo'] to be a list of dicts with keys
      like 'Report', 'Note ID', and 'Charttime'.
    - Selects the earliest entry by 'Charttime' when available; if no
      timestamps are available, falls back to the first list entry.
    - Output format: "Echocardiogram: <report>\n"
    """
    echo_list = action_results.get("Echo", [])
    if not isinstance(echo_list, list) or not echo_list:
        return "Echocardiogram: Not available.\n"

    with_time = []
    without_time = []
    for idx, e in enumerate(echo_list):
        ct = e.get("Charttime")
        has_time = (ct is not None) and (str(ct).lower() != "nat")
        (with_time if has_time else without_time).append((idx, e))

    if with_time:
        try:
            with_time.sort(key=lambda t: str(t[1].get("Charttime")))
        except Exception:
            with_time.sort(key=lambda t: t[0])
        earliest_entry = with_time[0][1]
    else:
        earliest_entry = without_time[0][1]

    report = str(earliest_entry.get("Report", "")).strip() or "Not available."
    return f"Echocardiogram: {report}\n"

# ----------------------------- Patient State (pure) -----------------------------

class _PurePatientState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.observations = {
            "physical_exam": None,
            "labs": [],
            "imaging": [],
            "ecg": [],
            "echocardiogram": [],
        }
        self.patient_history = None

    def add_observation(self, action: str, action_input: Any, observation: str):
        text = observation if isinstance(observation, str) else str(observation)
        def norm(s: str) -> str:
            return " ".join(s.split()).strip()
        n = norm(text)
        a = action.strip().lower()
        if a == "physical examination":
            cur = self.observations.get("physical_exam")
            if cur is None or norm(cur) != n:
                self.observations["physical_exam"] = text
        elif a == "laboratory tests":
            lst = self.observations.get("labs", [])
            if not any(norm(d.get("observation", "")) == n for d in lst):
                self.observations["labs"].append({"observation": text})
        elif a == "imaging":
            lst = self.observations.get("imaging", [])
            if not any(norm(d.get("observation", "")) == n for d in lst):
                self.observations["imaging"].append({"observation": text})
        elif a == "ecg":
            lst = self.observations.get("ecg", [])
            if not any(norm(d.get("observation", "")) == n for d in lst):
                self.observations["ecg"].append({"observation": text})
        elif a == "echocardiogram":
            lst = self.observations.get("echocardiogram", [])
            if not any(norm(d.get("observation", "")) == n for d in lst):
                self.observations["echocardiogram"].append({"observation": text})


_STATE = _PurePatientState()


def reset_patient_state():
    _STATE.reset()


def set_patient_history(text: Optional[str]):
    _STATE.patient_history = str(text) if text is not None else None


def update_patient_observation(action: str, action_input: Any, observation: str):
    _STATE.add_observation(action, action_input, observation)
