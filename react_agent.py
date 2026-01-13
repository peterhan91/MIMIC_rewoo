"""Pure-Python ReAct-style medical agent for MIMIC-IV.

This mirrors the core behavior of MIMIC-ReAct/agents without LangChain.
It uses the pure-Python tools provided in `tools.py`.

Key features:
- Strict tool schema with Thought / Action / Action Input loop
- Minimal, robust output parsing
- Patient State injection between steps
- Final Diagnosis is a dedicated tool call

Usage: see `run.py` (module name: react_agent).
"""

from __future__ import annotations

import json
import os
import pickle
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from thefuzz import process as fuzz_process

from tools import (
    physical_examination,
    lab_tests,
    imaging,
    ecg,
    echocardiogram,
    diagnostic_criteria,
    patient_state_tool,
    final_diagnosis,
    get_last_cot,
    reset_patient_state,
    set_patient_history,
    update_patient_observation,
)


# ----------------------------- Prompt Templates -----------------------------

DIAG_CRIT_TOOL_DESCR = (
    "\nDiagnostic Criteria: Examine the diagnostic criteria for a specific pathology. "
    "The pathology must be specified in the 'Action Input' field."
)

STOP_WORDS = ["Observation:", "Observations:", "observation:", "observations:"]


# ----------------------------- Minimal Utilities -----------------------------

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _sanitize_path_component(value: Optional[str], default: str = "patient") -> str:
    if not value:
        return default
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))
    cleaned = cleaned.strip("._-") or default
    return cleaned


def _message_content_to_text(content: Any) -> str:
    """Convert possibly structured chat message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                block_type = item.get("type")
                if block_type == "text":
                    parts.append(str(item.get("text", "")))
                elif block_type == "thinking":
                    thoughts = str(item.get("thinking", ""))
                    parts.append(f"[THINK]{thoughts}[/THINK]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if "content" in content:
            return _message_content_to_text(content.get("content"))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _split_hermes_think(text: str) -> Tuple[str, str]:
    """Split Hermes-style <think>...</think> from final content.

    Returns (thinking, final). If no think block, thinking="", final=text.
    If multiple, pick the last block and tail.
    """
    try:
        import re as _re
        s = str(text or "")
        matches = list(_re.finditer(r"<think>(.*?)</think>", s, flags=_re.DOTALL))
        if matches:
            m = matches[-1]
            return (m.group(1) or "").strip(), s[m.end():].strip()
        end_tag = "</think>"
        end_idx = s.rfind(end_tag)
        if end_idx != -1:
            return s[:end_idx].strip(), s[end_idx + len(end_tag):].strip()
        return "", s
    except Exception:
        return "", str(text or "")


def convert_labs_to_itemid(
    tests: List[str], lab_test_mapping_df: pd.DataFrame
) -> List[Union[int, str]]:
    """Minimal fuzzy mapping from names to itemids using labels and corresponding_ids.

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


# ----------------------------- Agent Components -----------------------------

LLMCallable = Callable[[str], str]


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[Any], str]


class DiagnosisParser:
    def __init__(self, allowed_tools: List[str], lab_test_mapping_df: Optional[pd.DataFrame] = None):
        self.allowed_tools = allowed_tools
        self.lab_test_mapping_df = lab_test_mapping_df

    def parse(self, llm_output: str) -> Tuple[str, Any, str, int]:
        """Parse Thought/Action/Action Input from the model output.

        Returns (action, action_input, thought_block, custom_parsings)
        """
        text = llm_output or ""
        custom = 0

        # Prefer JSON-structured output if present: {"thought": str, "action": str, "action_input": any}
        def _extract_first_json_obj(s: str) -> Optional[Dict[str, Any]]:
            s = s or ""
            # Try fenced JSON first
            m = re.search(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
            if m:
                inner = m.group(1)
                try:
                    obj = json.loads(inner)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
            # Scan for first balanced {} while ignoring braces inside strings
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
            # Fallback: whole string
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            return None

        j = _extract_first_json_obj(text)
        if isinstance(j, dict) and ("action" in j or "Action" in j):
            thought = str(j.get("thought") or j.get("Thought") or "").strip()
            action_raw = str(j.get("action") or j.get("Action") or "").strip()
            action = action_raw
            if "labs" in action.lower() or "blood work" in action.lower():
                action, custom = "Laboratory Tests", custom + 1
            if action not in self.allowed_tools:
                match, score = fuzz_process.extractOne(action, self.allowed_tools)
                if score >= 85:
                    action, custom = match, custom + (0 if score == 100 else 1)
            if action not in self.allowed_tools:
                raise ValueError(f"Invalid tool '{action_raw}'")

            ai = j.get("action_input")
            if action == "Imaging":
                if isinstance(ai, dict):
                    reg = str(ai.get("region", "")).strip()
                    mod = str(ai.get("modality", "")).strip()
                    if not (reg and mod):
                        raise ValueError("Imaging input must contain a region and modality")
                    ai = {"region": reg, "modality": mod}
                else:
                    ai = parse_imaging_action_input(str(ai or ""))
            elif action == "Laboratory Tests":
                if isinstance(ai, list):
                    parts = [str(p).strip() for p in ai if str(p).strip()]
                else:
                    parts = re.split(r",\s*(?![^()]*\))|\n", str(ai or ""))
                    parts = [p.strip() for p in parts if p and not p.isspace()]
                if self.lab_test_mapping_df is not None:
                    ai = convert_labs_to_itemid(parts, self.lab_test_mapping_df)
                else:
                    ai = parts
            elif action == "Diagnostic Criteria":
                if isinstance(ai, list):
                    ai = [str(p).strip() for p in ai if str(p).strip()]
                else:
                    parts = re.split(r",\s*(?![^()]*\))|\n| and ", str(ai or ""))
                    ai = [p.strip() for p in parts if p and not p.isspace()]
            elif action == "Patient State":
                ai = ai
            elif action == "Final Diagnosis":
                ai = ai
            return action, ai, thought, custom

        # Extract thought (everything until Action:)
        thought = ""
        m_th = re.search(r"Thought\s*:(.*?)(Action\s*:|$)", text, flags=re.DOTALL | re.IGNORECASE)
        if m_th:
            thought = m_th.group(1).strip()

        # Extract action
        m_act = re.search(r"Action\s*:\s*(.*?)(?:\n|$)", text, flags=re.IGNORECASE)
        if not m_act:
            raise ValueError("No Action found in model output")
        action_raw = m_act.group(1).strip()

        # Normalize known synonyms
        action = action_raw
        if "labs" in action.lower() or "blood work" in action.lower():
            action, custom = "Laboratory Tests", custom + 1
        if action not in self.allowed_tools:
            # Try fuzzy match against allowed tools
            match, score = fuzz_process.extractOne(action, self.allowed_tools)
            if score >= 85:
                action, custom = match, custom + (0 if score == 100 else 1)
        if action not in self.allowed_tools:
            raise ValueError(f"Invalid tool '{action_raw}'")

        # Extract action input (may be empty for some tools)
        ai = ""
        m_in = re.search(r"(Action\s+)?Input\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if m_in:
            ai = m_in.group(2).strip()
        # Convenience: treat 'None' as empty
        if ai.lower() == "none":
            ai, custom = "", custom + 1

        # Interpret action input
        if action == "Imaging":
            ai = parse_imaging_action_input(ai)
        elif action == "Laboratory Tests":
            # Convert to list by comma/newline
            parts = re.split(r",\s*(?![^()]*\))|\n", ai)
            parts = [p.strip() for p in parts if p and not p.isspace()]
            if self.lab_test_mapping_df is not None:
                ai = convert_labs_to_itemid(parts, self.lab_test_mapping_df)
            else:
                ai = parts
        elif action == "Diagnostic Criteria":
            parts = re.split(r",\s*(?![^()]*\))|\n| and ", ai)
            ai = [p.strip() for p in parts if p and not p.isspace()]
        elif action == "Patient State":
            # pass raw input (can be 'get' or JSON)
            ai = ai
        elif action == "Final Diagnosis":
            ai = ai

        return action, ai, thought, custom


class ReActAgent:
    def __init__(
        self,
        llm: Callable[[str], str],
        *,
        final_llm: Optional[Callable[[str], str]] = None,
        lab_test_mapping_df: Optional[pd.DataFrame] = None,
        provide_diagnostic_criteria: bool = False,
        include_ref_range: bool = False,
        bin_lab_results: bool = False,
        use_calculator: bool = False,
        ref_ranges: Optional[Dict[str, Any]] = None,
        calculator_critical_pct: float = 30.0,
        calculator_include_units: bool = True,
        stop_words: Optional[List[str]] = None,
        max_iterations: int = 10,
        tool_calling: bool = True,
        llama_ipython_mode: bool = False,
        include_function_defs_in_prompt: bool = False,
        system_prompt_override: Optional[Dict[str, Any]] = None,
        debug_print_messages: bool = False,
        debug_print_llm: bool = False,
        debug_save_messages: bool = False,
        debug_dir: Optional[str] = None,
        patient_state_pkl_path: Optional[str] = None,
        state_gather_only: bool = False,
        include_ddx_in_final: bool = False,
        final_reasoning_hint: str = "",
    ):
        self.llm = llm
        self.final_llm = final_llm or llm
        self.lab_test_mapping_df = lab_test_mapping_df
        self.llama_ipython_mode = bool(llama_ipython_mode)
        self.include_function_defs_in_prompt = bool(include_function_defs_in_prompt)
        self.provide_diagnostic_criteria = provide_diagnostic_criteria
        self.include_ref_range = include_ref_range
        self.bin_lab_results = bin_lab_results
        self.use_calculator = use_calculator
        self.ref_ranges = ref_ranges or {}
        self.calculator_critical_pct = calculator_critical_pct
        self.calculator_include_units = calculator_include_units
        self.stop_words = list((stop_words or []) + STOP_WORDS)
        self.max_iterations = max_iterations
        self.tool_calling = bool(tool_calling)
        self.debug_print_messages = bool(debug_print_messages)
        self.debug_print_llm = bool(debug_print_llm)
        self.debug_save_messages = bool(debug_save_messages)
        self.debug_dir = debug_dir
        self._system_prompt_override = deepcopy(system_prompt_override) if system_prompt_override else None
        self._debug_patient_dir: Optional[str] = None
        self._debug_case_index = 0
        self._patient_state_pkl_path = patient_state_pkl_path
        self._patient_state_cache: Dict[str, Any] = {}
        self._current_patient_id: Optional[Union[str, int]] = None
        self._state_gather_only = bool(state_gather_only)
        self.include_ddx_in_final = bool(include_ddx_in_final)
        self.final_reasoning_hint = str(final_reasoning_hint or "")
        if self._patient_state_pkl_path and os.path.exists(self._patient_state_pkl_path):
            try:
                with open(self._patient_state_pkl_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    self._patient_state_cache = loaded
            except Exception:
                self._patient_state_cache = {}

        # Request tracking for imaging repeat gating
        self._already_requested_scans: Dict[str, int] = {}
        # Track how many times we've nudged about specific lab N/A sets
        self._lab_reflexion_counts: Dict[Tuple[str, ...], int] = {}
        # Build tool inventory
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
        # HF tool-calling registry (name map) and tool definitions
        self._hf_tool_map: Dict[str, Callable[..., Any]] = {}
        self._hf_tool_defs: List[Dict[str, Any]] = []
        if self.tool_calling:
            self._register_hf_tools()

        # Parser (used for non-tool-calls and legacy ReAct steps)
        self.parser = DiagnosisParser(
            allowed_tools=list(self.tools.keys()),
            lab_test_mapping_df=self.lab_test_mapping_df,
        )

    def _register_default_tools(self):
        self.tools["Physical Examination"] = Tool(
            name="Physical Examination",
            description="Perform physical examination of patient and receive the observations.",
            run=lambda ai: physical_examination(self.patient),
        )
        self.tools["Laboratory Tests"] = Tool(
            name="Laboratory Tests",
            description=(
                "Run specific laboratory tests and receive their values. "
                "The specific tests must be specified in the 'Action Input' field."
            ),
            run=self._run_lab_tests,
        )
        self.tools["Imaging"] = Tool(
            name="Imaging",
            description=(
                "Do specific imaging scans and receive the radiologist report. "
                "Scan region AND modality must be specified in the 'Action Input' field."
            ),
            run=self._run_imaging,
        )
        # Cardiology tools
        self.tools["ECG"] = Tool(
            name="ECG",
            description="Retrieve the earliest available ECG report.",
            run=lambda ai: ecg(self.patient),
        )
        self.tools["Echocardiogram"] = Tool(
            name="Echocardiogram",
            description="Retrieve the earliest available Echocardiogram (Echo) report.",
            run=lambda ai: echocardiogram(self.patient),
        )
        if self.provide_diagnostic_criteria:
            self.tools["Diagnostic Criteria"] = Tool(
                name="Diagnostic Criteria",
                description=(
                    "Examine the diagnostic criteria for a specific pathology. "
                    "The pathology must be specified in the 'Action Input' field."
                ),
                run=lambda ai: diagnostic_criteria(ai),
            )
        self.tools["Patient State"] = Tool(
            name="Patient State",
            description="Read or update patient state (e.g., get or set leading diagnoses).",
            run=lambda ai: patient_state_tool(ai),
        )
        self.tools["Procedure Recommendation"] = Tool(
            name="Procedure Recommendation",
            description=(
                "Generate likely procedures based on patient state and final diagnosis, and map to ICD9/ICD10 codes. "
                "Returns JSON with queries and predicted codes."
            ),
            run=self._run_procedure,
        )
        self.tools["Medication Recommendation"] = Tool(
            name="Medication Recommendation",
            description=(
                "Review Medications on Admission and patient state to propose per-medication actions. "
                "Returns a JSON mapping medication name to {action, dose}."
            ),
            run=self._run_medication,
        )
        self.tools["Final Diagnosis"] = Tool(
            name="Final Diagnosis",
            description="Conclude with a final diagnosis JSON including rationale and diagnoses.",
            run=self._run_final_diagnosis,
        )

    # ----------------------------- Tool wrappers -----------------------------

    def _persist_patient_state(self) -> None:
        if not self._patient_state_pkl_path or self._current_patient_id is None:
            return
        try:
            state_raw = patient_state_tool("get")
        except Exception:
            return
        try:
            state = json.loads(state_raw)
        except Exception:
            state = state_raw
        pid = str(self._current_patient_id)
        self._patient_state_cache[pid] = state
        directory = os.path.dirname(self._patient_state_pkl_path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass
        try:
            with open(self._patient_state_pkl_path, "wb") as f:
                pickle.dump(self._patient_state_cache, f)
        except Exception:
            pass

    def _run_lab_tests(self, ai: List[Union[int, str]]) -> str:
        return lab_tests(
            action_results=self.patient,
            action_input=ai,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=self.include_ref_range,
            bin_lab_results=self.bin_lab_results,
            use_calculator=self.use_calculator,
            ref_ranges=self.ref_ranges,
            calculator_critical_pct=self.calculator_critical_pct,
            calculator_include_units=self.calculator_include_units,
            strict_compat=False,
        )

    def _run_imaging(self, ai: Dict[str, str]) -> str:
        return imaging(
            action_results=self.patient,
            action_input=ai,
            already_requested_scans=self._already_requested_scans,
        )

    def _run_medication(self, ai: Any) -> str:
        def _extract_notes(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                s = value.strip()
                if s.lower() in {"", "none", "null"}:
                    return ""
                return s
            if isinstance(value, dict):
                for key in ("notes", "note", "summary", "comment", "comments"):
                    if key in value:
                        try:
                            return str(value.get(key) or "").strip()
                        except Exception:
                            pass
                try:
                    return json.dumps(value, ensure_ascii=False)
                except Exception:
                    return str(value)
            if isinstance(value, (list, tuple, set)):
                return "\n".join([str(v).strip() for v in value if str(v).strip()])
            return str(value).strip()

        from tools import medication_recommendation as _med
        notes = _extract_notes(ai)
        return _med(self.final_llm, self.patient, notes=notes)

    def _run_procedure(self, ai: Any) -> str:
        # ai can be the final diagnosis JSON string or a plain final diagnosis string
        fd = ""
        if isinstance(ai, str):
            s = ai.strip()
            if s.startswith("{"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        fd = str(obj.get("final_diagnosis", "")).strip()
                    else:
                        fd = s
                except Exception:
                    fd = s
            else:
                fd = s
        elif isinstance(ai, dict):
            fd = str(ai.get("final_diagnosis", "")).strip()
        fd = fd or ""
        from tools import procedure_recommendation as _proc
        return _proc(self.final_llm, self.patient, final_diagnosis=fd)

    def _run_final_diagnosis(self, ai: Any) -> str:
        def _extract_notes(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                stripped = value.strip()
                lowered = stripped.lower()
                if lowered in {"", "none", "null"}:
                    return ""
                if (stripped.startswith("{") and stripped.endswith("}")) or (
                    stripped.startswith("[") and stripped.endswith("]")
                ):
                    try:
                        parsed = json.loads(stripped)
                    except Exception:
                        return stripped
                    return _extract_notes(parsed)
                return stripped
            if isinstance(value, dict):
                candidate_keys = (
                    "notes",
                    "note",
                    "summary",
                    "comment",
                    "comments",
                    "final_notes",
                )
                for key in candidate_keys:
                    if key in value:
                        candidate = _extract_notes(value.get(key))
                        if candidate:
                            return candidate
                try:
                    return json.dumps(value, ensure_ascii=False)
                except Exception:
                    return str(value)
            if isinstance(value, (list, tuple, set)):
                parts = []
                for item in value:
                    piece = _extract_notes(item)
                    if piece:
                        parts.append(piece)
                return "\n".join(parts)
            return str(value).strip()

        notes = _extract_notes(ai)

        self._persist_patient_state()
        if self._state_gather_only:
            try:
                return patient_state_tool("get")
            except Exception:
                return json.dumps({"state_gather_mode": True, "error": "Unable to serialize patient state"}, ensure_ascii=False)
        return final_diagnosis(
            llm=self.final_llm,
            notes=notes,
            include_ddx=self.include_ddx_in_final,
            reasoning_hint=self.final_reasoning_hint,
        )

    # ----------------------------- Core loop -----------------------------

    def _format_messages(self, patient_history: str) -> List[Dict[str, Any]]:
        tool_names = ", ".join(list(self.tools.keys()))
        tool_descr = (
            "The tools you can use are:\n\n"
            "Physical Examination: Perform physical examination of patient and receive the observations.\n"
            "Laboratory Tests: Run specific laboratory tests and receive their values. The specific tests must be specified in the 'Action Input' field.\n"
            "Imaging: Do specific imaging scans and receive the radiologist report. Scan region AND modality must be specified in the 'Action Input' field.\n"
            "ECG: Retrieve the earliest available ECG report.\n"
            "Echocardiogram: Retrieve the earliest available echocardiogram (Echo) report.\n"
            "Patient State: Read the current patient state or update the leading diagnoses using 'get' or {leading_differentials:[...]}.\n"
            "Procedure Recommendation: Generate likely procedures from state + final diagnosis; returns JSON with queries and predicted ICD9/10 codes.\n"
            "Medication Recommendation: Propose per-medication actions using patient state and Medications on Admission; returns JSON mapping medication -> {action, dose}.\n"
            "Final Diagnosis: When ready to conclude, call this to produce the diagnosis JSON with rationale and a diagnosis list."
        )
        if self.provide_diagnostic_criteria:
            tool_descr += DIAG_CRIT_TOOL_DESCR
        # Optionally inject formal function definitions into the prompt (Llama 3.2 lightweight zero-shot style)
        if self.include_function_defs_in_prompt and getattr(self, "_hf_tool_defs", None):
            try:
                fn_defs = [d.get("function", {}) for d in self._hf_tool_defs if isinstance(d, dict)]
                tool_descr += ("\n\nHere is a list of functions in JSON format that you can invoke:\n"
                               + json.dumps(fn_defs, ensure_ascii=False) + "\n")
                tool_descr += ("If you decide to invoke function(s), you may return them in the format: "
                               "[name1(arg1=value1, arg2=value2), name2(...)] with no other text.\n")
            except Exception:
                pass

        schema_line = '{"thought": string, "action": one of [' + tool_names + '], "action_input": value appropriate to the action}.\n\n'
        system_content = (
            "You are a medical artificial intelligence assistant. "
            "You give helpful, detailed and factually correct answers to help in clinical duties. "
            "Your goal is to correctly diagnose the patient. Always base your decisions on the patient state and observations.\n\n"
            + (
                "If your chat model supports function tool calls, prefer calling the appropriate function with arguments. "
                "If you cannot call a function, respond with EXACTLY ONE JSON object and no extra text: " + schema_line
                if self.tool_calling else
                "Respond with EXACTLY ONE JSON object and no extra text: " + schema_line
            )
            + (" Alternatively, you may respond with a single tool call JSON: {\"name\": function name, \"parameters\": {arg: value}} as supported by Huggingface custom tool calling.\n\n" if self.tool_calling else "")
            + "Keep all reasoning in 'thought'. The 'action' must be an exact tool name from [" + tool_names + "]. "
            + "Do not include curly braces { or } in 'thought'. Avoid JSON-like content except for the single required JSON object. "
            + "For Imaging, 'action_input' must be {region: <Abdomen|Chest|Head|Neck|Pelvis>, modality: <Ultrasound|CT|MRI|Radiograph>}. "
            + "For Laboratory Tests, 'action_input' must be a list of test names. For Patient State, use 'get' or {leading_differentials: [..]}.\n\n"
            + "To end the case, you MUST call the tool 'Final Diagnosis'. It returns {rationale, final_diagnosis, differential_diagnosis}. Do not write a final diagnosis directly; call the tool when ready.\n\n"
            + ("After your action, the tool result will be provided to you as a tool message.\n\n" if self.tool_calling else "After your action, the Observation will be provided to you.\n\n")
            + tool_descr + "\n\n"
            + "Rules:\n"
            + "- Do not invent action names outside [" + tool_names + "].\n"
            + "- If a tool is unavailable (e.g., imaging not available), continue reasoning based on currently gathered observations rather than switching to an unrelated condition.\n"
            + "- Before calling 'Final Diagnosis', carefully review the \"Current Patient State (auto)\" summary; if information is insufficient or inconsistent, collect more observations first.\n\n"
        )

        user_content = (
            "Consider the following case and come to a final diagnosis by thinking, planning, and using the aforementioned tools.\n\n"
            "Patient History: \n" + (patient_history or "") + "\n"
        )

        if self._system_prompt_override:
            system_message_candidate = deepcopy(self._system_prompt_override)
            if isinstance(system_message_candidate, dict):
                content = system_message_candidate.get("content")
                if isinstance(content, list):
                    augmented = list(content)
                    if system_content:
                        augmented.append({"type": "text", "text": system_content})
                    system_message_candidate["content"] = augmented
                elif isinstance(content, str):
                    joiner = "\n\n" if content and not content.endswith(("\n", " ")) else ""
                    system_message_candidate["content"] = content + (joiner + system_content if system_content else "")
                else:
                    system_message_candidate["content"] = system_content
                system_message_candidate.setdefault("role", "system")
                system_message = system_message_candidate
            else:
                system_message = {"role": "system", "content": system_content}
        else:
            system_message = {"role": "system", "content": system_content}

        return [
            system_message,
            {"role": "user", "content": user_content},
        ]

    def _call_llm_messages(self, messages: List[Dict[str, Any]], tools: Optional[List[Callable[..., Any]]] = None) -> str:
        try:
            return self.llm(messages, stop=self.stop_words, tools=tools)  # type: ignore[arg-type]
        except TypeError:
            # Fallback: convert to flat prompt
            sys_raw = next((m.get("content") for m in messages if m.get("role") == "system"), "")
            usr_raw = next((m.get("content") for m in messages if m.get("role") == "user"), "")
            sys = _message_content_to_text(sys_raw)
            usr = _message_content_to_text(usr_raw)
            prompt = (sys + "\n\n" + usr).strip()
            try:
                return self.llm(prompt, stop=self.stop_words)  # type: ignore[arg-type]
            except TypeError:
                return self.llm(prompt)


    def run_case(
        self,
        patient: Dict[str, Any],
        patient_history: str,
        patient_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """Run the ReAct loop over a single patient case until Final Diagnosis."""
        # Reset state
        reset_patient_state()
        set_patient_history(patient_history.strip() if patient_history else None)
        self.patient = patient
        self._already_requested_scans = {}
        self._lab_reflexion_counts = {}
        self._debug_case_index += 1
        self._current_patient_id = patient_id

        # Transcript stores tuples of (raw_llm, action_name, action_input, observation)
        transcript: List[Dict[str, Any]] = []

        # Build initial chat messages (system + case setup)
        messages = self._format_messages(patient_history)

        # Ensure debug dir exists if saving
        if self.debug_save_messages and self.debug_dir:
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
            except Exception:
                pass
            safe_id = f"case_{self._debug_case_index:03d}"
            if patient_id is not None:
                safe_component = _sanitize_path_component(str(patient_id), default="patient")
                safe_id = f"{self._debug_case_index:03d}_{safe_component}"
            patient_dir = os.path.join(self.debug_dir, safe_id)
            try:
                os.makedirs(patient_dir, exist_ok=True)
                self._debug_patient_dir = patient_dir
            except Exception:
                self._debug_patient_dir = self.debug_dir
        else:
            self._debug_patient_dir = None

        # Main iterations
        for step in range(self.max_iterations):
            # Inject a concise Patient State snapshot as a user turn
            try:
                state = json.loads(patient_state_tool("get"))
                state_for_llm = {
                    "patient_history": state.get("patient_history", None),
                    "observations": state.get("observations", {}),
                    "leading_differentials": state.get("leading_differentials", []),
                }
                messages.append({
                    "role": "user",
                    "content": f"Current Patient State (auto): {_safe_json(state_for_llm)}",
                })
            except Exception:
                pass

            # Generate next assistant turn
            if self.tool_calling and self._hf_tool_defs:
                llm_out = (self._call_llm_messages(messages, tools=self._hf_tool_defs) or "").strip()
                if self.debug_print_llm and llm_out:
                    print(f"[DEBUG] step {step} assistant out (tool-mode):\n{llm_out}")
                # Try to extract tool call when in tool-calling mode
                calls = self._extract_tool_calls(llm_out)
                if calls:
                    for call in calls:
                        tool_name = call.get("name")
                        arguments = call.get("arguments", {}) or {}
                        action_label = self._tool_label_from_name(tool_name)
                        messages.append({
                            "role": "assistant",
                            "tool_calls": [{"type": "function", "function": {"name": tool_name, "arguments": arguments}}],
                        })
                        try:
                            obs = self._dispatch_hf_tool(tool_name, arguments)
                            try:
                                update_patient_observation(action_label, arguments, obs)
                            except Exception:
                                pass
                        except Exception as e:
                            obs = f"Tool Error: {e}"
                            try:
                                update_patient_observation("Patient State", None, obs)
                            except Exception:
                                pass
                        if self.llama_ipython_mode:
                            payload = {"tool_call_id": tool_name, "output": str(obs)}
                            messages.append({"role": "ipython", "content": f"<|python_tag|>{json.dumps(payload, ensure_ascii=False)}<|eot_id|>"})
                        else:
                            messages.append({"role": "tool", "name": tool_name, "content": str(obs)})
                        if action_label == "Imaging":
                            self._maybe_inject_imaging_reflexion(messages, arguments, obs)
                        elif action_label == "Laboratory Tests":
                            self._maybe_inject_lab_planning_reflexion(messages, arguments)
                            self._maybe_inject_lab_reflexion(messages, arguments, obs)
                        self._debug_dump_messages(messages, step, tag=f"tool:{tool_name}", llm_out=llm_out)
                        transcript.append({
                            "raw": llm_out,
                            "action": action_label,
                            "action_input": arguments,
                            "observation": obs,
                        })
                        if str(tool_name).strip().lower() == "final_diagnosis":
                            if self._state_gather_only:
                                med_out = "Skipped: state gather mode"
                                proc_out = "Skipped: state gather mode"
                            else:
                                try:
                                    med_out = self._run_medication("")
                                except Exception:
                                    med_out = "Tool Error during medication recommendation"
                                try:
                                    proc_out = self._run_procedure(obs)
                                except Exception:
                                    proc_out = "Tool Error during procedure recommendation"
                            result = {
                                "output": obs,
                                "procedure_recommendation": proc_out,
                                "medication_recommendation": med_out,
                                "intermediate_steps": transcript,
                            }
                            if self._state_gather_only:
                                result["state_gather_mode"] = True
                            self._current_patient_id = None
                            return result
                    continue
                else:
                    # No tool call, treat as plain assistant content
                    messages.append({"role": "assistant", "content": llm_out})
            else:
                llm_out = (self._call_llm_messages(messages) or "").strip()
                if self.debug_print_llm and llm_out:
                    print(f"[DEBUG] step {step} assistant out:\n{llm_out}")
                messages.append({"role": "assistant", "content": llm_out})

            # Parse assistant content for ReAct-style tools/finalization
            try:
                action, action_input, thought, custom = self.parser.parse(llm_out)
            except Exception:
                invalid_tool_msg = (
                    f"Invalid action. Please choose a valid tool from {list(self.tools.keys())} "
                    f"or call 'Final Diagnosis' to end the case."
                )
                obs = invalid_tool_msg
                update_patient_observation("Patient State", None, obs)
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                self._debug_dump_messages(messages, step, tag="invalid", llm_out=llm_out)
                transcript.append({
                    "raw": llm_out,
                    "action": "Invalid",
                    "action_input": None,
                    "observation": obs,
                })
                continue

            # Execute tool or finalize
            if action == "Final Diagnosis":
                try:
                    obs = self.tools[action].run(action_input)
                except Exception as e:
                    obs = f"Tool Error: {e}"
                    try:
                        update_patient_observation("Patient State", None, obs)
                    except Exception:
                        pass
                    messages.append({"role": "user", "content": f"Observation: {obs}"})
                    self._debug_dump_messages(messages, step, tag="final-error", llm_out=llm_out)
                    transcript.append({
                        "raw": llm_out,
                        "action": action,
                        "action_input": action_input,
                        "observation": obs,
                    })
                    continue

                insuff = str(obs or "").strip().lower()
                if (
                    "insufficient data" in insuff and "gather more" in insuff
                ) or insuff in {
                    "insufficient data — gather more observations first",
                    "insufficient data - gather more observations first",
                }:
                    try:
                        update_patient_observation("Patient State", None, obs)
                    except Exception:
                        pass
                    messages.append({"role": "user", "content": f"Observation: {obs}"})
                    self._debug_dump_messages(messages, step, tag="final-insufficient", llm_out=llm_out)
                    transcript.append({
                        "raw": llm_out,
                        "action": action,
                        "action_input": action_input,
                        "observation": obs,
                    })
                    continue

                # Finalize
                transcript.append({
                    "raw": llm_out,
                    "action": action,
                    "action_input": action_input,
                    "observation": obs,
                })
                self._debug_dump_messages(messages, step, tag="final", llm_out=llm_out)
                # Post-final: also perform Medication Recommendation and Procedure Recommendation
                if self._state_gather_only:
                    med_out = "Skipped: state gather mode"
                    proc_out = "Skipped: state gather mode"
                else:
                    try:
                        med_out = self._run_medication("")
                    except Exception:
                        med_out = "Tool Error during medication recommendation"
                    try:
                        proc_out = self._run_procedure(obs)
                    except Exception:
                        proc_out = "Tool Error during procedure recommendation"
                result = {
                    "output": obs,
                    "procedure_recommendation": proc_out,
                    "medication_recommendation": med_out,
                    "intermediate_steps": transcript,
                }
                # Attach CoT if captured by tools
                try:
                    result["final_diagnosis_cot"] = get_last_cot("final_diagnosis").get("thinking", "")
                    result["medication_recommendation_cot"] = get_last_cot("medication_recommendation").get("thinking", "")
                    result["procedure_recommendation_cot"] = get_last_cot("procedure_recommendation").get("thinking", "")
                except Exception:
                    pass
                if self._state_gather_only:
                    result["state_gather_mode"] = True
                self._current_patient_id = None
                return result

            # Non-final tools
            try:
                obs = self.tools[action].run(action_input)
            except Exception as e:
                obs = f"Tool Error: {e}"
                try:
                    update_patient_observation("Patient State", None, obs)
                except Exception:
                    pass
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                self._debug_dump_messages(messages, step, tag="tool-error", llm_out=llm_out)
                transcript.append({
                    "raw": llm_out,
                    "action": action,
                    "action_input": action_input,
                    "observation": obs,
                })
                continue

            try:
                update_patient_observation(action, action_input, obs)
            except Exception:
                pass

            # In tool-calling mode, map certain actions to tool role messages
            if self.tool_calling and action in {"Physical Examination", "Imaging", "Laboratory Tests", "Patient State", "ECG", "Echocardiogram", "Medication Recommendation", "Procedure Recommendation"}:
                # Synthesize a tool call message and tool result for consistency
                if action == "Physical Examination":
                    tname = "physical_examination"
                    args = {}
                elif action == "Imaging":
                    tname = "imaging"
                    args = action_input if isinstance(action_input, dict) else {}
                elif action == "Laboratory Tests":
                    tname = "laboratory_tests"
                    args = {"tests": action_input if isinstance(action_input, list) else []}
                elif action == "ECG":
                    tname = "ecg"
                    args = {}
                elif action == "Echocardiogram":
                    tname = "echocardiogram"
                    args = {}
                elif action == "Procedure Recommendation":
                    tname = "procedure_recommendation"
                    args = {"final_diagnosis": action_input if isinstance(action_input, str) else ""}
                elif action == "Medication Recommendation":
                    tname = "medication_recommendation"
                    args = {"notes": action_input if isinstance(action_input, str) else ""}
                else:
                    # Patient State
                    if isinstance(action_input, dict) and action_input.get("leading_differentials"):
                        tname = "patient_state_update"
                        args = {"leading_differentials": action_input.get("leading_differentials", [])}
                    else:
                        tname = "patient_state_get"
                        args = {}
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": tname, "arguments": args}}],
                })
                if self.llama_ipython_mode:
                    payload = {"tool_call_id": tname, "output": str(obs)}
                    messages.append({"role": "ipython", "content": f"<|python_tag|>{json.dumps(payload, ensure_ascii=False)}<|eot_id|>"})
                else:
                    messages.append({"role": "tool", "name": tname, "content": str(obs)})
            else:
                # Non-tool-calling flow: append observation as user message
                messages.append({"role": "user", "content": f"Observation: {obs.strip()}"})
                if action == "Imaging":
                    self._maybe_inject_imaging_reflexion(messages, action_input, obs)
                elif action == "Laboratory Tests":
                    self._maybe_inject_lab_planning_reflexion(messages, action_input)
                    self._maybe_inject_lab_reflexion(messages, action_input, obs)
            self._debug_dump_messages(messages, step, tag=action, llm_out=llm_out)
            transcript.append({
                "raw": llm_out,
                "action": action,
                "action_input": action_input,
                "observation": obs,
            })

        # If loop exits without final diagnosis
        result = {
            "output": "Max iterations reached without Final Diagnosis.",
            "intermediate_steps": transcript,
        }
        if self._state_gather_only:
            self._persist_patient_state()
            try:
                state_snapshot = patient_state_tool("get")
            except Exception:
                state_snapshot = result["output"]
            result["output"] = state_snapshot
            result["procedure_recommendation"] = "Skipped: state gather mode"
            result["medication_recommendation"] = "Skipped: state gather mode"
            result["state_gather_mode"] = True
        self._current_patient_id = None
        return result

    # ----------------------------- HF Tool-calling helpers -----------------------------

    def _register_hf_tools(self) -> None:
        # Always register the three high-ROI tools when tool-calling is enabled
        def _fn_physical_examination() -> str:
                """Perform physical examination of patient and receive the observations."""
                # Use the registered tool runner to avoid shadowing the imported name
                return self.tools["Physical Examination"].run(None)

        def imaging(region: str, modality: str) -> str:
                """Do specific imaging scans and receive the radiologist report.

                Args:
                    region: The anatomical region to scan (e.g., Abdomen, Chest, Head, Neck, Pelvis)
                    modality: The imaging modality (e.g., Ultrasound, CT, MRI, Radiograph)
                Returns:
                    The radiologist report string for the requested scan.
                """
                ai = {"region": str(region), "modality": str(modality)}
                return self._run_imaging(ai)

        def laboratory_tests(tests: List[str]) -> str:
                """Run specific laboratory tests and receive their values.

                Args:
                    tests: A list of lab test names (e.g., ["WBC", "CRP"]).
                Returns:
                    Formatted lab results string for the requested tests.
                """
                clean = [str(t).strip() for t in (tests or []) if str(t).strip()]
                ai: List[Union[int, str]]
                if self.lab_test_mapping_df is not None:
                    ai = convert_labs_to_itemid(clean, self.lab_test_mapping_df)
                else:
                    ai = clean
                return self._run_lab_tests(ai)

        def ecg() -> str:
                """Retrieve the earliest available ECG report."""
                return self.tools["ECG"].run(None)

        def echocardiogram() -> str:
                """Retrieve the earliest available Echocardiogram (Echo) report."""
                return self.tools["Echocardiogram"].run(None)

        def patient_state_get() -> str:
                """Get the current Patient State as JSON (history, observations, diagnoses)."""
                return patient_state_tool("get")

        def patient_state_update(leading_differentials: List[str]) -> str:
                """Update the leading diagnoses in the Patient State.

                Args:
                    leading_differentials: A list of leading diagnoses (strings).
                Returns:
                    "OK" on success or an error message.
                """
                diffs = [str(d).strip() for d in (leading_differentials or []) if str(d).strip()]
                return patient_state_tool({"leading_differentials": diffs})

        def final_diagnosis(notes: str = "", **kwargs: Any) -> str:
                """Conclude the case with a diagnosis JSON.

                Args:
                    notes: Optional brief notes to consider in the finalization.
                    **kwargs: Additional optional parameters (e.g., overrides) passed by the model.
                Returns:
                    A JSON string with keys: rationale, final_diagnosis, differential_diagnosis.
                """
                if kwargs:
                    payload: Dict[str, Any] = dict(kwargs)
                    payload["notes"] = notes
                    return self._run_final_diagnosis(payload)
                return self._run_final_diagnosis(notes)

        def procedure_recommendation(final_diagnosis: str) -> str:
                """Generate likely procedures and map to ICD9/ICD10 using an index.

                Args:
                    final_diagnosis: The predicted final diagnosis string.
                Returns:
                    A JSON string with keys: queries, predicted.icd9, predicted.icd10.
                """
                return self._run_procedure(final_diagnosis)

        def medication_recommendation(notes: str = "", **kwargs: Any) -> str:
                """Review Medications on Admission and propose per-medication actions.

                Args:
                    notes: Optional brief notes to consider (e.g., allergies, contraindications).
                Returns:
                    A JSON string mapping medication name -> {action, dose}.
                """
                if kwargs:
                    payload: Dict[str, Any] = dict(kwargs)
                    payload["notes"] = notes
                    return self._run_medication(payload)
                return self._run_medication(notes)

        self._hf_tool_map.update({
            "physical_examination": _fn_physical_examination,
            "imaging": imaging,
            "laboratory_tests": laboratory_tests,
            "ecg": ecg,
            "echocardiogram": echocardiogram,
            "patient_state_get": patient_state_get,
            "patient_state_update": patient_state_update,
            "procedure_recommendation": procedure_recommendation,
            "medication_recommendation": medication_recommendation,
            "final_diagnosis": final_diagnosis,
        })

        # Build tool definition schemas for the chat template
        self._hf_tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "physical_examination",
                    "description": "Perform physical examination of patient and receive the observations.",
                    "parameters": {"type": "object", "required": [], "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "imaging",
                    "description": "Do specific imaging scans and receive the radiologist report.",
                    "parameters": {
                        "type": "object",
                        "required": ["region", "modality"],
                        "properties": {
                            "region": {
                                "type": "string",
                                "description": "Anatomical region to scan.",
                                "enum": ["Abdomen", "Chest", "Head", "Neck", "Pelvis"],
                            },
                            "modality": {
                                "type": "string",
                                "description": "Imaging modality to use.",
                                "enum": ["Ultrasound", "CT", "MRI", "Radiograph"],
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ecg",
                    "description": "Retrieve the earliest available ECG report.",
                    "parameters": {"type": "object", "required": [], "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "echocardiogram",
                    "description": "Retrieve the earliest available Echocardiogram (Echo) report.",
                    "parameters": {"type": "object", "required": [], "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "laboratory_tests",
                    "description": (
                        "Run specific laboratory tests and receive their values. "
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["tests"],
                        "properties": {
                            "tests": {
                                "type": "array",
                                "description": "List of lab test names to order (strings).",
                                "minItems": 1,
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "patient_state_get",
                    "description": "Get the current Patient State as JSON (history, observations, leading diagnoses).",
                    "parameters": {"type": "object", "required": [], "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "patient_state_update",
                    "description": "Update the leading diagnoses in the Patient State.",
                    "parameters": {
                        "type": "object",
                        "required": ["leading_differentials"],
                        "properties": {
                            "leading_differentials": {
                                "type": "array",
                                "description": "Leading diagnoses.",
                                "minItems": 1,
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "procedure_recommendation",
                    "description": (
                        "Generate likely procedures based on patient state and final diagnosis; map to ICD9/ICD10 codes. "
                        "Returns JSON with queries and predicted codes."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["final_diagnosis"],
                        "properties": {
                            "final_diagnosis": {
                                "type": "string",
                                "description": "The predicted final diagnosis string.",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "final_diagnosis",
                    "description": (
                        "Conclude the case with a diagnosis JSON. "
                        "Returns JSON with rationale, final_diagnosis, differential_diagnosis."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": [],
                        "properties": {
                            "notes": {
                                "type": "string",
                                "description": "Optional brief notes to consider in finalization.",
                                "default": "",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "medication_recommendation",
                    "description": (
                        "Review Medications on Admission and patient state to propose per-medication actions. "
                        "Returns JSON mapping medication name -> {action, dose}."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": [],
                        "properties": {
                            "notes": {
                                "type": "string",
                                "description": "Optional brief notes to consider (e.g., allergies).",
                                "default": "",
                            }
                        },
                    },
                },
            },
        ]

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract one or more tool calls from model text.

        Supports:
        - Single JSON object: {"name": ..., "arguments": {...}} or {"name": ..., "parameters": {...}}
        - OpenAI-style: {"tool_calls": [{"type": "function", "function": {"name":..., "arguments": {...}}}]}
        - Hermes 4: <tool_call> { ... } </tool_call> blocks (ChatML special tokens)
        - Llama 3.3 bracketed syntax: [func(arg='v'), func2(...)] or single func(arg=...)
        Returns a list of {"name": str, "arguments": dict}.
        """
        s = text or ""
        out: List[Dict[str, Any]] = []

        # 0) Hermes 4: <tool_call> ... </tool_call>
        try:
            for mtag in re.finditer(r"<tool_call[^>]*>(.*?)</tool_call>", s, flags=re.DOTALL | re.IGNORECASE):
                inner = (mtag.group(1) or "").strip()
                if not inner:
                    continue
                # Extract first JSON object inside the tag
                mj = re.search(r"\{.*\}", inner, flags=re.DOTALL)
                if not mj:
                    continue
                blob = mj.group(0)
                try:
                    obj = json.loads(blob)
                except Exception:
                    continue
                # Accept either {name, arguments} or {function: {name, arguments}}
                if isinstance(obj, dict):
                    if "name" in obj and ("arguments" in obj or "parameters" in obj):
                        args = obj.get("arguments") or obj.get("parameters") or {}
                        if not isinstance(args, dict):
                            args = {}
                        out.append({"name": str(obj.get("name")), "arguments": args})
                    elif obj.get("type") == "function" and isinstance(obj.get("function"), dict):
                        fn = obj.get("function")
                        name = fn.get("name")
                        args = fn.get("arguments") or fn.get("parameters") or {}
                        if not isinstance(args, dict):
                            args = {}
                        if name:
                            out.append({"name": str(name), "arguments": args})
            if out:
                return out
        except Exception:
            pass

        # 1) JSON object {name, arguments}
        m = re.search(r"\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\"arguments\"\s*:\s*(\{.*?\})[^{}]*\}", s, flags=re.DOTALL)
        if m:
            name = m.group(1)
            args_str = m.group(2)
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}
            out.append({"name": name, "arguments": args})
            return out

        # 2) JSON object {name, parameters} (Llama 3.1 custom)
        m_params = re.search(r"\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\"parameters\"\s*:\s*(\{.*?\})[^{}]*\}", s, flags=re.DOTALL)
        if m_params:
            name = m_params.group(1)
            args_str = m_params.group(2)
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}
            out.append({"name": name, "arguments": args})
            return out

        # 3) tool_calls wrapper
        m2 = re.search(r"\"tool_calls\"\s*:\s*\[(.*?)\]", s, flags=re.DOTALL)
        if m2:
            inner = m2.group(1)
            # crude iteration over possible {"function": {"name":..., "arguments": {...}}}
            for mfc in re.finditer(r"\{[^{}]*\"function\"\s*:\s*\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\"arguments\"\s*:\s*(\{.*?\})[^{}]*\}\s*\}", inner, flags=re.DOTALL):
                name = mfc.group(1)
                args_str = mfc.group(2)
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                out.append({"name": name, "arguments": args})
            if out:
                return out

        # 4) Llama 3.3 bracketed: [fn(a=1, b='x'), fn2(...)] OR single fn(a=1)
        calls = self._parse_bracketed_calls(s)
        if calls:
            return calls

        return []

    # ------------------ Llama 3.3 bracketed function-call parser ------------------
    def _parse_bracketed_calls(self, s: str) -> List[Dict[str, Any]]:
        s = (s or "").strip()
        # Extract content between [ ... ] if present; else use entire string
        content = s
        m = re.search(r"\[(.*)\]", s, flags=re.DOTALL)
        if m:
            content = m.group(1)

        # Split top-level function calls by commas not inside parentheses or quotes
        parts = []
        buf = []
        depth = 0
        in_str = False
        q = ''
        esc = False
        for ch in content:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == q:
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    q = ch
                    buf.append(ch)
                elif ch == '(':
                    depth += 1
                    buf.append(ch)
                elif ch == ')':
                    depth = max(0, depth - 1)
                    buf.append(ch)
                elif ch == ',' and depth == 0:
                    part = ''.join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                else:
                    buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)

        def parse_call(expr: str) -> Optional[Dict[str, Any]]:
            expr = expr.strip()
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*", expr, flags=re.DOTALL)
            if not m:
                return None
            name = m.group(1)
            argstr = m.group(2).strip()
            args = self._parse_kwargs(argstr)
            return {"name": name, "arguments": args}

        out: List[Dict[str, Any]] = []
        for p in parts:
            call = parse_call(p)
            if call and call.get("name"):
                out.append(call)
        return out

    def _parse_kwargs(self, s: str) -> Dict[str, Any]:
        """Parse keyword arg string like: key='v', n=1, flag=true, items=[1,'a']"""
        s = (s or "").strip()
        if not s:
            return {}
        items = []
        buf = []
        depth = 0
        in_str = False
        q = ''
        esc = False
        for ch in s:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == q:
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    q = ch
                    buf.append(ch)
                elif ch in '([':
                    depth += 1
                    buf.append(ch)
                elif ch in ')]':
                    depth = max(0, depth - 1)
                    buf.append(ch)
                elif ch == ',' and depth == 0:
                    part = ''.join(buf).strip()
                    if part:
                        items.append(part)
                    buf = []
                else:
                    buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            items.append(last)

        def parse_value(v: str) -> Any:
            v = v.strip()
            # Strip quotes
            if (len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'"))):
                return v[1:-1]
            # Booleans/None
            low = v.lower()
            if low == 'true':
                return True
            if low == 'false':
                return False
            if low == 'none' or low == 'null':
                return None
            # Numbers
            try:
                if re.fullmatch(r"[-+]?\d+", v):
                    return int(v)
                if re.fullmatch(r"[-+]?\d*\.\d+", v) or re.fullmatch(r"[-+]?\d+\.\d*", v):
                    return float(v)
            except Exception:
                pass
            # Lists (simple)
            if len(v) >= 2 and v[0] == '[' and v[-1] == ']':
                inner = v[1:-1]
                # Split list items
                vals = []
                b2 = []
                d2 = 0
                ins = False
                qq = ''
                esc2 = False
                for ch2 in inner:
                    if ins:
                        b2.append(ch2)
                        if esc2:
                            esc2 = False
                        elif ch2 == '\\':
                            esc2 = True
                        elif ch2 == qq:
                            ins = False
                    else:
                        if ch2 in ('"', "'"):
                            ins = True
                            qq = ch2
                            b2.append(ch2)
                        elif ch2 in '([':
                            d2 += 1
                            b2.append(ch2)
                        elif ch2 in ')]':
                            d2 = max(0, d2 - 1)
                            b2.append(ch2)
                        elif ch2 == ',' and d2 == 0:
                            it = ''.join(b2).strip()
                            if it:
                                vals.append(parse_value(it))
                            b2 = []
                        else:
                            b2.append(ch2)
                it = ''.join(b2).strip()
                if it:
                    vals.append(parse_value(it))
                return vals
            # Fallback string
            return v

        out: Dict[str, Any] = {}
        for it in items:
            if '=' in it:
                k, v = it.split('=', 1)
                out[str(k).strip()] = parse_value(v)
        return out

    def _dispatch_hf_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        fn = self._hf_tool_map.get(str(name).strip())
        if not fn:
            raise ValueError(f"Unknown tool: {name}")
        # Call with kwargs if applicable
        try:
            return fn(**arguments)  # type: ignore[arg-type]
        except TypeError:
            return fn()  # type: ignore[misc]

    def _tool_label_from_name(self, name: str) -> str:
        n = str(name or "").strip().lower()
        if n == "physical_examination":
            return "Physical Examination"
        if n == "imaging":
            return "Imaging"
        if n == "laboratory_tests":
            return "Laboratory Tests"
        if n == "ecg":
            return "ECG"
        if n == "echocardiogram":
            return "Echocardiogram"
        if n.startswith("patient_state"):
            return "Patient State"
        if n == "procedure_recommendation":
            return "Procedure Recommendation"
        if n == "medication_recommendation":
            return "Medication Recommendation"
        if n == "final_diagnosis":
            return "Final Diagnosis"
        return n or "Tool"

    def _maybe_inject_imaging_reflexion(
        self,
        messages: List[Dict[str, Any]],
        action_input: Any,
        observation: Any,
    ) -> None:
        text = str(observation or "")
        if "Not available" not in text:
            return
        if not isinstance(action_input, dict):
            return
        region = str(action_input.get("region", "") or "").strip()
        modality = str(action_input.get("modality", "") or "").strip()
        if region and modality:
            scan_name = f"{region} {modality}"
        elif region:
            scan_name = f"{region} imaging"
        elif modality:
            scan_name = f"{modality} imaging"
        else:
            scan_name = "requested imaging"
        guidance = (
            f"Reflexion: The {scan_name} was unavailable. "
            "Consider selecting another appropriate imaging modality for the same diagnostic question "
            "or explain why no further imaging is required before proceeding."
        )
        if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == guidance:
            return
        messages.append({"role": "user", "content": guidance})

    def _maybe_inject_lab_reflexion(
        self,
        messages: List[Dict[str, Any]],
        action_input: Any,
        observation: Any,
    ) -> None:
        text = str(observation or "")
        # Capture lines that explicitly returned N/A (case-insensitive)
        na_items = re.findall(r"^\s*(?:\([^)]*\)\s*)?([^:]+):\s*N/A\b", text, flags=re.MULTILINE | re.IGNORECASE)
        # Normalize for deduping but keep display names
        cleaned: List[str] = []
        seen = set()
        for item in na_items:
            name = item.strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            cleaned.append(name)
        if not cleaned:
            return
        key = tuple(sorted(s.lower() for s in cleaned))
        count = self._lab_reflexion_counts.get(key, 0)
        if count >= 2:
            return
        self._lab_reflexion_counts[key] = count + 1
        base_hint = "These lab requests returned N/A: " + ", ".join(cleaned) + "."
        extra_hint = ""
        follow_up = " Consider alternative or more specific tests based on your clinical reasoning."
        guidance = f"Reflexion: {base_hint}{follow_up}"
        if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == guidance:
            return
        messages.append({"role": "user", "content": guidance})

    def _maybe_inject_lab_planning_reflexion(
        self,
        messages: List[Dict[str, Any]],
        action_input: Any,
    ) -> None:
        if not isinstance(action_input, (dict, list)):
            return
        if isinstance(action_input, dict):
            tests = action_input.get("tests", [])
        else:
            tests = action_input
        if tests is None:
            tests = []
        if not isinstance(tests, list):
            tests = [tests]
        guidance = (
            "Reflexion: Double-check that your lab order covers the key questions (inflammation, infection, organ function) "
            "and consider bundling complementary tests instead of single isolated labs."
        )
        if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == guidance:
            return
        messages.append({"role": "user", "content": guidance})

    # ----------------------------- Debug helpers -----------------------------

    def _debug_dump_messages(self, messages: List[Dict[str, Any]], step: int, tag: str, llm_out: Optional[str] = None) -> None:
        if self.debug_print_llm and llm_out:
            thk, fin = _split_hermes_think(llm_out)
            print(f"[DEBUG] step {step} llm_out:\n{fin or llm_out}")
        if self.debug_print_messages:
            try:
                print(f"[DEBUG] step {step} messages ({tag}):\n" + json.dumps(messages, ensure_ascii=False, indent=2))
            except Exception:
                print(f"[DEBUG] step {step} messages ({tag}): <unprintable>")
        base_debug_dir = self._debug_patient_dir or self.debug_dir
        if self.debug_save_messages and base_debug_dir:
            try:
                path = os.path.join(base_debug_dir, f"messages_step_{step:02d}_{tag}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
