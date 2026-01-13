"""Pure Python implementations of project tools (no internal imports).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
import json
import time
import re
from thefuzz import process as fuzz_process
import copy
import logging


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

# ------------------------ Diagnostic Criteria (pure) ------------------------

_CRITERIA_MAP: Dict[str, str] = {
    "appendicitis": (
        "To diagnose appendicitis consider the following criteria: General symptoms usually include pain around the naval "
        "that shifts to the right lower quadrant (RLQ) of the abdomen, accompanied by fever and nausea or vomiting. During "
        "a physical examination, a patient might show RLQ tenderness, positive rebound tenderness, or signs of peritonitis. "
        "Laboratory tests may reveal signs of an inflammatory response, such as an elevated white blood cell count and "
        "elevated C-reactive protein levels. Imaging may disclose an enlarged appendix or possibly an appendicolith."
    ),
    "cholecystitis": (
        "To diagnose cholecystitis, consider the following criteria: General symptoms usually include pain in the right upper "
        "quadrant (RUQ) of the abdomen, fever, and nausea. During a physical examination, a patient might display RUQ tenderness "
        "or indications of jaundice. Laboratory tests may reveal signs of inflammation, such as elevated white blood cell count "
        "and C-reactive protein levels, liver damage, indicated through heightened Alanine Aminotransferase (ALT) or Asparate "
        "Aminotransferase (AST) levels, or gallbladder damage, indicated through heightened Bilirubin or Gamma Glutamyltransferase "
        "levels. Imaging may show gallstones, thickened gallbladder walls, pericholecystic fluid, and a distended gallbladder."
    ),
    "diverticulitis": (
        "To diagnose diverticulitis consider the following criteria: General symptoms typically encompass abdominal pain, "
        "primarily in the left lower quadrant (LLQ), along with fever, and nausea or vomiting. During a physical examination, "
        "a patient may display tenderness in the LLQ, fever, and signs of peritonitis. Laboratory tests often reveal signs of "
        "inflammation and infection, which may include an elevated white blood cell count and elevated C-reactive protein levels. "
        "Imaging findings often include bowel wall thickening, diverticula, inflammation, or abscesses around the affected "
        "segment of the colon."
    ),
    "pancreatitis": (
        "To diagnose pancreatitis consider the following criteria: General symptoms usually include abdominal pain, primarily "
        "in the epigastric region, along with nausea or vomiting. During a physical examination, a patient might display "
        "epigastric tenderness, fever, and signs of jaundice. Laboratory tests may reveal signs of inflammation, such as "
        "elevated white blood cell count and C-reactive protein levels, and pancreatic damage, indicated through heightened "
        "Amylase or Lipase levels. Further lab tests of hematocrit, urea nitrogen, triglycerides, calcium, sodium and potassium "
        "can indicate the severity of the disease. Imaging may show inflammation of the pancreas or fluid collection."
    ),
}


def _closest_key(name: str, keys: List[str]) -> str:
    s = name.strip().lower()
    if not s:
        return ""
    if fuzz_process is None:
        # Simple fallback: exact or startswith
        for k in keys:
            if s == k:
                return k
        for k in keys:
            if s in k or k in s:
                return k
        return s
    match = fuzz_process.extractOne(s, keys)
    if match and match[1] >= 80:
        return match[0]
    return s


def diagnostic_criteria(action_input: Union[List[str], Dict[str, Any]]) -> str:
    """Return diagnostic criteria text for requested pathologies (with header).

    Accepts list[str] or dict with {'pathologies': [...]}. Fuzzy matches names.
    Writes missing names to 'no_diagnostic_criteria.txt' for parity.
    """
    if isinstance(action_input, dict):
        names = action_input.get("pathologies", [])
        if isinstance(names, str):
            names = [names]
    elif isinstance(action_input, list):
        names = action_input
    else:
        names = [str(action_input)]

    out_lines: List[str] = []
    keys = list(_CRITERIA_MAP.keys())
    for raw in names:
        key = _closest_key(str(raw), keys)
        text = _CRITERIA_MAP.get(key)
        if not text:
            out_lines.append(f"Diagnostic criteria for {raw} is not available.\n")
            try:
                with open("no_diagnostic_criteria.txt", "a") as f:
                    f.write(f"{raw}\n")
            except Exception:
                pass
        else:
            out_lines.append(text + "\n")
    return "Diagnostic Criteria:\n" + "".join(out_lines)


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
            "diagnostic_criteria": [],
        }
        self.patient_history = None
        self.leading_differentials: List[str] = []
        self.final_diagnosis = None
        self.differential_diagnosis: List[str] = []
        self.last_access_ts: Optional[float] = None
        # Expert diagnosis artifacts
        self._ddx_top_list: List[str] = []  # stored under key "diagnosis"
        self._ddx_expert_map: Dict[str, str] = {}  # stored under key "expert DDx"
        self._ddx_expert_reasoning: Optional[str] = None  # stored under key "expert reasoning"

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
        elif a == "diagnostic criteria":
            lst = self.observations.get("diagnostic_criteria", [])
            if not any(norm(d.get("observation", "")) == n for d in lst):
                self.observations["diagnostic_criteria"].append({"observation": text})

    def set_leading_differentials(self, diffs: List[str]):
        clean = []
        for d in diffs:
            s = str(d).strip()
            if s and len(s) <= 80:
                clean.append(s)
        self.leading_differentials = clean[:5]

    def set_final_diagnosis(self, text: Optional[str], differentials: Optional[List[str]] = None):
        """Persist the most recent confirmed final diagnosis and diagnoses."""
        s = str(text or "").strip()
        self.final_diagnosis = s if s else None
        if differentials is not None:
            clean: List[str] = []
            for d in differentials:
                val = str(d).strip()
                if val:
                    clean.append(val[:160])
            self.differential_diagnosis = clean[:10]
        # Clear stored diagnoses list if explicitly passing empty list
        elif self.final_diagnosis is None:
            self.differential_diagnosis = []

    def mark_accessed(self):
        self.last_access_ts = time.time()

    def get_state(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "patient_history": self.patient_history,
            "leading_differentials": list(self.leading_differentials),
            "final_diagnosis": self.final_diagnosis,
            "differential_diagnosis": list(self.differential_diagnosis),
            # Additional diagnosis artifacts requested
            "diagnosis": list(self._ddx_top_list),
            "expert DDx": dict(self._ddx_expert_map),
            "expert reasoning": (self._ddx_expert_reasoning or ""),
        }

    # New helpers for diagnosis persistence
    def set_differential_list(self, diffs: List[str]):
        clean: List[str] = []
        for d in diffs:
            s = str(d).strip()
            if s:
                clean.append(s[:160])
        self.differential_diagnosis = clean[:10]

    def set_ddx_results(self, top_list: List[str], expert_map: Dict[str, str], expert_reasoning: Optional[str]):
        # Store exact keys as requested; also keep internal normalized copies
        tl: List[str] = []
        for d in (top_list or []):
            s = str(d).strip()
            if s:
                tl.append(s[:160])
        self._ddx_top_list = tl[:10]
        emap: Dict[str, str] = {}
        for k, v in (expert_map or {}).items():
            kk = str(k).strip()
            vv = str(v or "").strip()
            if kk and vv:
                emap[kk[:160]] = vv[:1000]
        # Restrict map keys to top_list when available
        if self._ddx_top_list:
            filtered: Dict[str, str] = {}
            for dx in self._ddx_top_list:
                if dx in emap:
                    filtered[dx] = emap[dx]
            emap = filtered if filtered else emap
        self._ddx_expert_map = emap
        self._ddx_expert_reasoning = str(expert_reasoning or "").strip() or None


_STATE = _PurePatientState()
# Store last chain-of-thought (CoT) snippets for key tools
_LAST_COT: Dict[str, Dict[str, str]] = {}


def _set_last_cot(name: str, thinking: str, final_text: str, raw: str) -> None:
    try:
        _LAST_COT[str(name).lower()] = {
            "thinking": str(thinking or ""),
            "final": str(final_text or ""),
            "raw": str(raw or ""),
        }
    except Exception:
        pass


def get_last_cot(name: Optional[str] = None) -> Dict[str, Any]:
    """Return last captured CoT for a tool name or all.

    Keys: thinking, final, raw. Empty dict if unavailable.
    """
    try:
        if not name:
            return copy.deepcopy(_LAST_COT)
        return copy.deepcopy(_LAST_COT.get(str(name).lower(), {}))
    except Exception:
        return {}


def _split_hermes_think(text: str) -> Tuple[str, str]:
    """Split <think>...</think> from the remainder of the text.

    Returns (thinking, final). If not found, returns ("", text).
    """
    try:
        import re as _re
        s = str(text or "")
        matches = list(_re.finditer(r"<think>(.*?)</think>", s, flags=_re.DOTALL))
        if matches:
            m = matches[-1]
            think = (m.group(1) or "").strip()
            final = s[m.end():].strip()
            return think, final
        # Fallback: only closing tag present (e.g., some Qwen templates)
        end_tag = "</think>"
        end_idx = s.rfind(end_tag)
        if end_idx != -1:
            think = s[:end_idx].strip()
            final = s[end_idx + len(end_tag):].strip()
            return think, final
        return "", s
    except Exception:
        return "", str(text or "")

# Keep the last vote details for final_diagnosis so the agent can log them
_LOG = logging.getLogger(__name__)


def _log_tool_input(name: str, payload: Any) -> None:
    if not _LOG.isEnabledFor(logging.INFO):
        return
    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except Exception:
        serialized = repr(payload)
    _LOG.info("%s input: %s", name, serialized)


def reset_patient_state():
    _STATE.reset()


def set_patient_history(text: Optional[str]):
    _STATE.patient_history = str(text) if text is not None else None


def update_patient_observation(action: str, action_input: Any, observation: str):
    _STATE.add_observation(action, action_input, observation)


def patient_state_tool(action_input: Union[str, Dict[str, Any]]) -> str:
    if isinstance(action_input, str):
        s = action_input.strip().lower()
        if s in {"get", "read", "review", "show", "view"} or "get state" in s:
            _STATE.mark_accessed()
            return json.dumps(_STATE.get_state())
        if action_input.strip().startswith("{"):
            try:
                action_input = json.loads(action_input)
            except Exception:
                pass
    if isinstance(action_input, dict) and "leading_differentials" in action_input:
        diffs = action_input.get("leading_differentials", [])
        if not isinstance(diffs, list) or not diffs:
            return "Patient State Error: 'leading_differentials' must be a non-empty list of strings."
        _STATE.set_leading_differentials([str(d) for d in diffs])
        _STATE.mark_accessed()
        return "OK"
    return (
        "Patient State Error: Invalid input. Use 'get' or JSON like "
        "{\"leading_differentials\":[\"appendicitis\"]}."
    )


# ------------------------------ Final Diagnosis (pure) ------------------------------

def _fd_system_text(reasoning_hint: str = "") -> str:
    return (
        "Act as an experienced emergency doctor. "
        "Use ONLY the Patient State (JSON) provided below and your medical knowledge to determine the final diagnosis and a patient-specific treatment plan. "
        "Respond with EXACTLY ONE JSON object and no extra text using this schema: "
        '{"final_diagnosis": string, "treatment": string, "confidence": number (0-1, optional), "rationale": string (optional), "differential_diagnosis": [string, ...]}\n\n'
        f"{reasoning_hint}\n\n"
    )


def final_diagnosis(
    llm,
    notes: str = "",
    tags: Optional[Dict[str, str]] = None,  # kept for API compatibility; ignored
    reasoning_hint: str = "",
    include_ddx: bool = False,
) -> str:
    state = _STATE.get_state()
    state_for_llm = {
        "patient_history": state.get("patient_history", None),
        "observations": state.get("observations", {}),
        "leading_differentials": state.get("leading_differentials", []),
    }
    if include_ddx:
        # Optionally enrich with prior diagnosis artifacts
        dx_list = state.get("diagnosis") or []
        expert_map = state.get("expert DDx") or {}
        expert_refl = state.get("expert reasoning") or ""
        if dx_list:
            state_for_llm["diagnosis"] = dx_list
        if expert_map:
            state_for_llm["expert DDx"] = expert_map
        if expert_refl:
            state_for_llm["expert reasoning"] = expert_refl
    ph = state_for_llm.get("patient_history")
    obs = state_for_llm.get("observations", {}) or {}
    if not (ph and str(ph).strip()) and not any([
        bool(obs.get("physical_exam") and str(obs.get("physical_exam")).strip()),
        len(obs.get("labs") or []) > 0,
        len(obs.get("imaging") or []) > 0,
        len(obs.get("diagnostic_criteria") or []) > 0,
    ]):
        return "Insufficient data — gather more observations first"

    # Build messages for the main final diagnosis draft
    system_msg = _fd_system_text(reasoning_hint or "")
    user_msg = (
        "Patient State (JSON):\n" + json.dumps(state_for_llm, ensure_ascii=False) + "\n\n"
        + "Additional Notes (optional): " + (notes or "") + "\n"
    )

    draft = (llm([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ], stop=[]) or "").strip()
    # Capture CoT (if present)
    try:
        thk, fin = _split_hermes_think(draft)
        _set_last_cot("final_diagnosis", thk, fin, draft)
    except Exception:
        pass

    def _extract_first_json_obj(s: str) -> Optional[Dict[str, Any]]:
        s = s or ""
        # Fenced JSON first
        m = re.search(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            inner = m.group(1)
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # Balanced braces ignoring strings
        depth = 0
        start = None
        in_str = False
        str_ch = ''
        escape = False
        for i, ch in enumerate(s):
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == str_ch:
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    str_ch = ch
                elif ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start is not None:
                            chunk = s[start:i + 1]
                            try:
                                obj = json.loads(chunk)
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:
                                pass
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return None

    def _normalize_diff_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            out: List[str] = []
            for item in value:
                s_item = str(item).strip()
                if s_item:
                    out.append(s_item[:160])
            return out[:10]
        if isinstance(value, str):
            s_val = value.strip()
            if not s_val:
                return []
            if s_val.startswith("[") and s_val.endswith("]"):
                try:
                    parsed = json.loads(s_val)
                    return _normalize_diff_list(parsed)
                except Exception:
                    return []
            return [s_val[:160]]
        try:
            return _normalize_diff_list(list(value))
        except Exception:
            return []

    def _extract_final_diagnosis_text(text: str, _depth: int = 0) -> Optional[str]:
        if _depth > 3:
            return None
        s_txt = str(text or "").strip()
        if not s_txt:
            return None
        try:
            decoded = json.loads(s_txt)
        except Exception:
            decoded = None
        if isinstance(decoded, dict):
            val = decoded.get("final_diagnosis")
            if isinstance(val, str) and val.strip():
                return val.strip()
        elif isinstance(decoded, str) and decoded != s_txt:
            inner = _extract_final_diagnosis_text(decoded, _depth=_depth + 1)
            if inner:
                return inner
        m = re.search(r'"final_diagnosis"\s*:\s*"([^"]+)"', s_txt)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return candidate
        return None

    def _extract_differential_list(text: str, _depth: int = 0) -> List[str]:
        if _depth > 3:
            return []
        s_txt = str(text or "").strip()
        if not s_txt:
            return []
        try:
            decoded = json.loads(s_txt)
        except Exception:
            decoded = None
        if isinstance(decoded, dict):
            vals = _normalize_diff_list(decoded.get("differential_diagnosis"))
            if vals:
                return vals
        elif isinstance(decoded, (list, tuple, set)):
            vals = _normalize_diff_list(decoded)
            if vals:
                return vals
        elif isinstance(decoded, str) and decoded != s_txt:
            inner = _extract_differential_list(decoded, _depth=_depth + 1)
            if inner:
                return inner
        m = re.search(r'"differential_diagnosis"\s*:\s*(\[[^\]]*\])', s_txt, flags=re.DOTALL)
        if m:
            arr = m.group(1)
            try:
                parsed = json.loads(arr)
                return _normalize_diff_list(parsed)
            except Exception:
                pass
        return []

    obj = _extract_first_json_obj(draft)
    if isinstance(obj, dict) and obj:
        try:
            fd = obj.get("final_diagnosis")
            diffs = _normalize_diff_list(obj.get("differential_diagnosis"))
            _STATE.set_final_diagnosis(fd, diffs)
        except Exception:
            pass
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
    else:
        fd_candidate = _extract_final_diagnosis_text(draft)
        diff_candidate = _extract_differential_list(draft)
        if fd_candidate or diff_candidate:
            try:
                _STATE.set_final_diagnosis(fd_candidate, diff_candidate if diff_candidate else None)
            except Exception:
                pass
    return draft


# ------------------------------ Diagnoses (pure) ------------------------------

def _ddx_system_text_top10(reasoning_hint: str = "") -> str:
    return (
        "Act as an experienced clinician. Use ONLY the Patient State (JSON) provided below. "
        "List the top 10 most likely diagnoses for this case. "
        "Respond with EXACTLY ONE JSON array of 10 strings; each string is a precise diagnosis name. "
        "Do not invent tests or findings that are not present.\n\n"
        f"{reasoning_hint}\n"
    )


def _ddx_system_text_rationales() -> str:
    return (
        "For each provided diagnosis, write a concise rationale grounded ONLY in the Patient State (JSON). "
        "Return EXACTLY ONE JSON object mapping diagnosis -> rationale (1–3 sentences each). "
        "Do not add extra keys or diagnoses beyond the list."
    )


def _ddx_system_text_reflection() -> str:
    return (
        "Write a short expert reflection on the overall diagnosis list: its coverage, the most plausible paths, "
        "and key discriminating evidence still needed. Keep it to 3–5 sentences."
    )


def _parse_bullets_as_list(text: str, max_n: int = 10) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # Accept formats like '- Foo', '1) Foo', '1. Foo'
        s = re.sub(r"^\s*[-*+\d\.)]+\s*", "", s)
        s = s.strip()
        if not s:
            continue
        if s.lower().startswith("the top 10"):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def generate_differential_diagnosis(
    llm,
    notes: str = "",
    reasoning_hint: str = "",
) -> str:
    """Generate diagnoses: top-10 list, per-diagnosis rationales, and a short reflection.

    Stores results into patient state under keys: 'diagnosis', 'expert DDx', 'expert reasoning'.
    Also updates 'differential_diagnosis' (top-10) and 'leading_differentials' (top-5).
    Returns a JSON string with these three keys.
    """
    state = _STATE.get_state()
    state_for_llm = {
        "patient_history": state.get("patient_history", None),
        "observations": state.get("observations", {}),
        "leading_differentials": state.get("leading_differentials", []),
    }

    # 1) Top-10 diagnoses
    sys_top = _ddx_system_text_top10(reasoning_hint or "")
    usr_top = (
        "Case: " + json.dumps(state_for_llm, ensure_ascii=False) + "\n\n"
        + "What are the top 10 most likely diagnoses? Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10).\n"
        + "The top 10 diagnoses are:\n"
        + "Output as a JSON array of 10 strings.\n"
        + ("Additional Notes: " + notes if notes else "")
    )
    _log_tool_input("Diagnosis (Top-10): system prompt", {"content": sys_top})
    _log_tool_input("Diagnosis (Top-10): user input", {"content": usr_top})
    raw_top = (llm([
        {"role": "system", "content": sys_top},
        {"role": "user", "content": usr_top},
    ], stop=[]) or "").strip()
    try:
        thk, fin = _split_hermes_think(raw_top)
        _set_last_cot("differential_diagnosis_top10", thk, fin, raw_top)
    except Exception:
        pass
    # Prefer fenced/inline JSON array; fallback to bullets
    try:
        top_list = _extract_first_json_array(raw_top)  # type: ignore[name-defined]
    except Exception:
        top_list = None
    if not top_list:
        top_list = _parse_bullets_as_list(raw_top, max_n=10)
    # Normalize, dedupe, cap at 10
    norm_top: List[str] = []
    seen = set()
    for d in top_list or []:
        s = str(d).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        norm_top.append(s)
        if len(norm_top) >= 10:
            break

    # 2) Per-diagnosis rationales
    expert_map: Dict[str, str] = {}
    if norm_top:
        sys_rat = _ddx_system_text_rationales()
        usr_rat = (
            "Patient State (JSON):\n" + json.dumps(state_for_llm, ensure_ascii=False) + "\n\n"
            + "Diagnoses (JSON array):\n" + json.dumps(norm_top, ensure_ascii=False) + "\n\n"
            + "Return ONE JSON object mapping each diagnosis to a concise rationale (1–3 sentences)."
        )
        _log_tool_input("Diagnosis (Rationales): user input", {"content": usr_rat})
        raw_rat = (llm([
            {"role": "system", "content": sys_rat},
            {"role": "user", "content": usr_rat},
        ], stop=[]) or "").strip()
        try:
            thk, fin = _split_hermes_think(raw_rat)
            _set_last_cot("differential_diagnosis_rationales", thk, fin, raw_rat)
        except Exception:
            pass
        # Parse first JSON object
        def _first_obj(s: str) -> Optional[Dict[str, Any]]:
            m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
            if m:
                try:
                    o = json.loads(m.group(1))
                    if isinstance(o, dict):
                        return o
                except Exception:
                    pass
            try:
                o2 = json.loads(s)
                return o2 if isinstance(o2, dict) else None
            except Exception:
                return None
        obj_rat = _first_obj(raw_rat) or {}
        # Restrict to provided diagnoses
        for dx in norm_top:
            val = obj_rat.get(dx)
            if isinstance(val, str) and val.strip():
                expert_map[dx] = val.strip()

    # 3) Short reflection
    expert_reflection = ""
    sys_ref = _ddx_system_text_reflection()
    usr_ref = (
        "Patient State (JSON):\n" + json.dumps(state_for_llm, ensure_ascii=False) + "\n\n"
        + "Diagnoses (JSON array):\n" + json.dumps(norm_top, ensure_ascii=False) + "\n\n"
        + "Write the short expert reflection."
    )
    raw_ref = (llm([
        {"role": "system", "content": sys_ref},
        {"role": "user", "content": usr_ref},
    ], stop=[]) or "").strip()
    try:
        thk, fin = _split_hermes_think(raw_ref)
        _set_last_cot("differential_diagnosis_reflection", thk, fin, raw_ref)
        expert_reflection = fin or raw_ref
    except Exception:
        expert_reflection = raw_ref

    # Persist into patient state
    try:
        if norm_top:
            _STATE.set_differential_list(norm_top)
            _STATE.set_leading_differentials(norm_top[:5])
        _STATE.set_ddx_results(norm_top, expert_map, expert_reflection)
    except Exception:
        pass

    out = {
        "diagnosis": norm_top,
        "expert DDx": expert_map,
        "expert reasoning": expert_reflection,
    }
    try:
        return json.dumps(out, ensure_ascii=False)
    except Exception:
        return str(out)


# ------------------------------ Medications (pure) ------------------------------

def _med_system_text(reasoning_hint: str = "") -> str:
    example = r'''Example:
Patient summary: Acute appendicitis treated with laparoscopic appendectomy; patient on simvastatin and naproxen at home.
Admission meds:
{"Simvastatin": "40 mg QHS", "Naproxen": "500 mg daily"}
Output:
{
  "Acetaminophen": {"action": "new", "dose": "650 mg Q6H PRN pain"},
  "Oxycodone": {"action": "new", "dose": "5 mg Q4H PRN pain"},
  "Docusate sodium": {"action": "new", "dose": "100 mg BID"},
  "Polyethylene glycol 3350": {"action": "new", "dose": "17 g daily PRN constipation"},
  "Simvastatin": {"action": "continue", "dose": "n/a"},
  "Naproxen": {"action": "continue", "dose": "n/a"}
}'''
    return (
        "You are a clinical pharmacist responsible for preparing a complete outpatient discharge medication plan.\n"
        "Consider and analyse ONLY the Patient State (JSON) and Medications on Admission provided.\n"
        "Your plan should include:\n"
        "1. All ongoing home medications that should be continued, held, change dose, or stopped.\n"
        "2. Any NEW medications required based on the current illness, procedure, or hospital course.\n"
        "3. Correct actions and doses.\n\n"
        "OUTPUT:\n"
        "Respond with EXACTLY ONE JSON object. Each key is a medication name (generic preferred). Each value is an object with fields:\n"
        "- action: one of [new, continue, hold, stop, change_dose]\n"
        "- dose: 'n/a' unless action is 'new' or 'change_dose', in which case provide a concrete dose string (e.g., '100 mg BID').\n\n"
        f"{reasoning_hint}\n\n"
        f"{example}\n"
    )



def medication_recommendation(
    llm,
    action_results: Dict[str, Any],
    notes: str = "",
    reasoning_hint: str = "",
) -> str:
    state = _STATE.get_state()
    diff_list = state.get("differential_diagnosis") or state.get("leading_differentials") or []
    state_for_llm = {
        "patient_history": state.get("patient_history", None),
        "observations": state.get("observations", {}),
        "final_diagnosis": state.get("final_diagnosis", None),
        "differential_diagnosis": diff_list,
    }
    meds_on_adm = action_results.get("Medications on Admission", "Not available.")

    system_msg = _med_system_text(reasoning_hint or "")
    user_msg = (
        "Patient State (JSON):\n" + json.dumps(state_for_llm, ensure_ascii=False) + "\n\n"
        + "Medications on Admission:\n" + (json.dumps(meds_on_adm, ensure_ascii=False) if not isinstance(meds_on_adm, str) else meds_on_adm) + "\n\n"
    )

    _log_tool_input(
        "Medication Recommendation: system prompt",
        {"content": system_msg},
    )

    _log_tool_input(
        "Medication Recommendation: user input",
        {"content": user_msg},
    )

    draft = (llm([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ], stop=[]) or "").strip()
    # Capture CoT (if present)
    try:
        thk, fin = _split_hermes_think(draft)
        _set_last_cot("medication_recommendation", thk, fin, draft)
    except Exception:
        pass

    def _extract_first_json_obj(s: str) -> Optional[Dict[str, Any]]:
        s = s or ""
        m = re.search(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            inner = m.group(1)
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        depth = 0
        start = None
        in_str = False
        str_ch = ''
        escape = False
        for i, ch in enumerate(s):
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == str_ch:
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    str_ch = ch
                elif ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start is not None:
                            chunk = s[start:i + 1]
                            try:
                                obj = json.loads(chunk)
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:
                                pass
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return None

    obj = _extract_first_json_obj(draft)
    if isinstance(obj, dict) and obj:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
    return draft


# ------------------------------ Procedures (pure) ------------------------------

def _proc_system_text() -> str:
    return (
        "You are a clinical coder with expert knowledge of ICD-9 and ICD-10 procedure terminology. "
        "Using ONLY the Patient State (JSON) and the Final Diagnosis provided, list canonical THERAPEUTIC or "
        "INVASIVE procedures most likely performed during this patient's admission. "
        "If no procedure required, just return None."
        "Return EXACTLY ONE JSON object with the key 'queries' and value as a list. "
        "Do NOT include diagnostic or imaging procedures.\n\n"
        '''Examples:

{
  "queries": [
    "Endoscopic sphincterotomy and papillotomy",
    "Endoscopic removal of stone(s) from biliary tract"
  ]
}

{
  "queries": [
    "Implant of pulsation balloon",
    "Percutaneous transluminal coronary angioplasty [PTCA]",
    "Insertion of non-drug-eluting coronary artery stent(s)",
    "Left heart cardiac catheterization",
    "Coronary arteriography using two catheters"
  ]
}

{
  "queries": [
    "None"
  ]
}
'''
    )


def _extract_first_json_array(s: str) -> Optional[List[str]]:
    s = s or ""
    # Fenced JSON first
    m = re.search(r"```json\s*(\[.*?\])\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Bracket scan
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch in ('"', "'"):
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
            elif ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        chunk = s[start:i + 1]
                        try:
                            arr = json.loads(chunk)
                            if isinstance(arr, list):
                                return [str(x).strip() for x in arr if str(x).strip()]
                        except Exception:
                            pass
    return None


def _load_icd_index(index_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            idx = json.load(f)
        return idx if isinstance(idx, dict) else None
    except Exception:
        return None


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _match_queries_to_index(queries: List[str], index: Dict[str, Any], fuzzy_threshold: int = 90, top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    def build_catalog(entries: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        cat: List[Tuple[str, Dict[str, Any]]] = []
        for e in entries or []:
            title = str(e.get('title') or '').strip()
            if title:
                cat.append((_normalize_text(title), e))
            syns = e.get('synonyms') or []
            if isinstance(syns, list):
                for syn in syns:
                    syn_s = str(syn or '').strip()
                    if syn_s:
                        cat.append((_normalize_text(syn_s), e))
        return cat

    icd9_entries = index.get('icd9') or []
    icd10_entries = index.get('icd10') or []
    icd9_cat = build_catalog(icd9_entries)
    icd10_cat = build_catalog(icd10_entries)

    def match_one(q: str, catalog: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        qn = _normalize_text(q)
        # Exact matches first
        exact = [e for t, e in catalog if t == qn]
        results: List[Dict[str, Any]] = []
        seen_codes = set()
        for e in exact:
            code = e.get('code')
            key = ('code', str(code))
            if key in seen_codes:
                continue
            seen_codes.add(key)
            results.append({"code": e.get('code'), "title": e.get('title')})
            if len(results) >= top_k:
                return results
        # Fuzzy match
        if fuzz_process and catalog:
            choices = [t for t, _ in catalog]
            extracted = fuzz_process.extract(qn, choices, limit=max(top_k * 3, 5))
            for cand, score in extracted or []:
                if score < fuzzy_threshold:
                    continue
                # Grab the first entry with this normalized text
                for t, e in catalog:
                    if t == cand:
                        code = e.get('code')
                        key = ('code', str(code))
                        if key in seen_codes:
                            continue
                        seen_codes.add(key)
                        results.append({"code": e.get('code'), "title": e.get('title')})
                        break
                if len(results) >= top_k:
                    break
        return results

    out_icd9: List[Dict[str, Any]] = []
    out_icd10: List[Dict[str, Any]] = []
    seen9 = set()
    seen10 = set()
    for q in queries:
        for item in match_one(q, icd9_cat):
            code = item.get('code')
            key = ('code', str(code))
            if key in seen9:
                continue
            seen9.add(key)
            # Normalize ICD9 code to int when possible
            try:
                if isinstance(code, str) and code.isdigit():
                    item['code'] = int(code)
            except Exception:
                pass
            out_icd9.append(item)
        for item in match_one(q, icd10_cat):
            code = item.get('code')
            key = ('code', str(code))
            if key in seen10:
                continue
            seen10.add(key)
            # Ensure ICD10 code remains string
            item['code'] = str(item.get('code'))
            out_icd10.append(item)

    return {"icd9": out_icd9, "icd10": out_icd10}


def procedure_recommendation(
    llm,
    action_results: Dict[str, Any],
    final_diagnosis: str,
    index_path: str = "icd_procedures_index.json",
    max_queries: int = 6,
    fuzzy_threshold: int = 90,
    top_k: int = 1,
) -> str:
    # Build minimal, leakage-free state for LLM
    state = _STATE.get_state()
    diff_list = state.get("differential_diagnosis") or state.get("leading_differentials") or []
    state_for_llm = {
        "patient_history": state.get("patient_history", None),
        "observations": state.get("observations", {}),
        "final_diagnosis": state.get("final_diagnosis", None),
        "differential_diagnosis": diff_list,
    }

    # Ask LLM for canonical procedure names
    sys_msg = _proc_system_text()
    usr_msg = (
        "Patient State (JSON):\n" + json.dumps(state_for_llm, ensure_ascii=False) + "\n\n"
        + f"Return procedure names as a JSON array.\n"
    )

    _log_tool_input(
        "Procedure Recommendation: user message",
        {"content": usr_msg},
    )

    raw = (llm([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": usr_msg},
    ], stop=[]) or "").strip()
    # Capture CoT (if present)
    try:
        thk, fin = _split_hermes_think(raw)
        _set_last_cot("procedure_recommendation", thk, fin, raw)
    except Exception:
        pass
    queries = _extract_first_json_array(raw) or []

    # Load index and map
    index = _load_icd_index(index_path) or {"icd9": [], "icd10": []}
    predicted = _match_queries_to_index(queries, index, fuzzy_threshold=fuzzy_threshold, top_k=top_k)

    out = {
        "queries": queries,
        "predicted": {
            "icd9": predicted.get("icd9", []),
            "icd10": predicted.get("icd10", []),
        },
    }
    try:
        return json.dumps(out, ensure_ascii=False)
    except Exception:
        return str(out)
# ------------------------------ Convenience Map ------------------------------

def get_pure_tools() -> Dict[str, Any]:
    return {
        "Physical Examination": physical_examination,
        "Laboratory Tests": lab_tests,
        "Imaging": imaging,
        "ECG": ecg,
        "Echocardiogram": echocardiogram,
        "Diagnostic Criteria": diagnostic_criteria,
        "Patient State": patient_state_tool,
        "Diagnoses": generate_differential_diagnosis,
        "Final Diagnosis": final_diagnosis,
        "Medication Recommendation": medication_recommendation,
        "Procedure Recommendation": procedure_recommendation,
    }
