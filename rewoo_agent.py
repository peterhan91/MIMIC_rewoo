"""ReWOO-style medical agent for MIMIC-IV.

This agent decouples planning from observation:
  1) Planner: generates a compact step-by-step plan with explicit tool calls
     using the format:
        Plan: <description>
        #E1 = <Tool>[<Input>]
        Plan: <description>
        #E2 = <Tool>[<Input using #E1 if needed>]
        ...
  2) Worker: executes the planned tool calls, resolves #E references, and
     collects observations.
  3) Solver: produces a single diagnosis string from patient history + evidence,
     mirroring the STEER zero-shot style.

Focus: maximize diagnostic accuracy while maintaining efficiency.
  - One LLM call for planning (vs. iterative ReAct)
  - Finalization via a single-pass diagnosis generator

Usage: wired into run.py via --agent-type rewoo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import pandas as pd

from tools import (
    physical_examination,
    lab_tests,
    imaging,
    ecg,
    echocardiogram,
    reset_patient_state,
    set_patient_history,
    update_patient_observation,
)

# Reuse robust helpers from the ReAct agent
from rewoo_helpers import (
    convert_labs_to_itemid,
    parse_imaging_action_input,
)


# ----------------------------- Planner prompt -----------------------------

def _tool_descriptions() -> str:
    lines: List[str] = []
    lines.append("Tools (choose minimally; prefer high-yield steps):")
    lines.append("Physical Examination[input]: Perform physical examination; input can be left blank.")
    lines.append("Laboratory Tests[input]: Order specific tests, provide a comma-separated list. Choose targeted tests for the suspected organ system(s).")
    lines.append("Imaging[input]: Do specific imaging; input must specify region and modality.")
    lines.append("ECG[input]: Retrieve the ECG report; only ordering when cardiac symptoms/signs are present.")
    lines.append("Echocardiogram[input]: Retrieve the echocardiogram (Echo) report; only ordering when clinically indicated.")
    lines.append("\nImaging Regions: Abdomen, Chest, Head, Neck, Pelvis")
    lines.append("Imaging Modalities: Ultrasound, CT, MRI, Radiograph")
    return "\n".join(lines)


def _planner_system_prompt() -> str:
    return (
        "You are an experienced clinician. Using your medical knowledge and the patient’s presentation, propose a focused and efficient plan for evidence gathering that ensures diagnostic precision and minimizes unnecessary tests.\n"
        "Return ONLY a sequence of lines using this schema:\n\n"
        "Plan: <concise rationale for next step>\n"
        "#E1 = <Tool>[<Input>]\n"
        "Plan: <next step>\n"
        "#E2 = <Tool>[<Input possibly informed by #E1>]\n"
        "...\n\n"
        "Rules:\n"
        "- Minimize the number of steps, but ensure you gather sufficient information for an accurate and safe final diagnosis.\n"
        "- Prefer high-yield, patient-condition specific steps.\n"
        "- Use the exact tool names and input formats described below.\n"
        "- Do not conclude with a diagnosis here; your output is ONLY the plan.\n\n"
        "- Output only Plan/#E lines; do not add headers, bullets, or code fences.\n\n"
        + _tool_descriptions()
    )


def _planner_user_prompt(patient_history: str) -> str:
    return (
        "Patient History:\n" + (patient_history or "") + "\n\n"
        "Begin! Output only Plan/#E lines as specified. Keep it concise and tailored.\n"
    )


def _solver_system_prompt() -> str:
    return (
        "You are a medical artificial intelligence assistant. You directly diagnose patients based on the provided information "
        "to assist a doctor in clinical duties. Your goal is to correctly diagnose the patient. Based on the provided information "
        "you will provide the final diagnosis of the most severe pathology. Don't write any further information. "
        "Give only a single diagnosis."
    )


def _solver_user_prompt(patient_history: str, evidence: str) -> str:
    ev = (evidence or "").strip() or "None."
    return (
        "Patient History:\n" + (patient_history or "") + "\n\n"
        "Evidence:\n" + ev + "\n\n"
        "Final Diagnosis:"
    )


def _normalize_final_diagnosis(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    s = re.sub(r"^(final diagnosis|diagnosis|primary diagnosis|main diagnosis)\\s*:\\s*", "", s, flags=re.I)
    line = s.splitlines()[0].strip()
    return line


# ----------------------------- Utilities -----------------------------
def _split_hermes_think(text: str) -> Tuple[str, str]:
    """Split Hermes-style <think>...</think> from final content.

    Returns (thinking, final). If no think block, thinking="", final=text.
    If multiple think blocks, returns the last one and the tail after it.
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
        end_tag = "</think>"
        end_idx = s.rfind(end_tag)
        if end_idx != -1:
            return s[:end_idx].strip(), s[end_idx + len(end_tag):].strip()
        return "", s
    except Exception:
        return "", str(text or "")

def _extract_tool_and_input(raw: str) -> Tuple[str, Optional[str]]:
    """Parse '<Tool>[<Input>]' or '<Tool>' into components."""
    s = (raw or "").strip()
    if "[" in s and s.endswith("]"):
        tool, rest = s.split("[", 1)
        tool = tool.strip()
        arg = rest[:-1].strip()
        return tool, arg
    return s.strip(), None


def _parse_keyvals(arg: str) -> Dict[str, str]:
    """Parse 'k=v, k2=v2' into a dict."""
    out: Dict[str, str] = {}
    if not arg:
        return out
    parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))|\n", arg) if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[str(k).strip()] = str(v).strip()
    return out


def _split_csv(arg: str) -> List[str]:
    return [p.strip() for p in re.split(r",\s*(?![^()]*\))|\n", str(arg or "")) if p.strip()]


_TOOL_ALIASES = {
    "physical exam": "Physical Examination",
    "physical examination": "Physical Examination",
    "pe": "Physical Examination",
    "lab tests": "Laboratory Tests",
    "labs": "Laboratory Tests",
    "laboratory tests": "Laboratory Tests",
    "imaging": "Imaging",
    "ecg": "ECG",
    "ekg": "ECG",
    "echocardiogram": "Echocardiogram",
    "echo": "Echocardiogram",
}


def _normalize_tool_name(raw: str) -> str:
    name = str(raw or "").strip()
    if not name:
        return name
    key = name.lower()
    return _TOOL_ALIASES.get(key, name)


def _render_tool_call(raw_tool_call: str, resolved_input: Optional[str]) -> str:
    tool_name, arg = _extract_tool_and_input(raw_tool_call)
    if resolved_input is not None:
        return f"{tool_name}[{resolved_input}]"
    return raw_tool_call


# ----------------------------- Agent -----------------------------

@dataclass
class PlanResult:
    raw: str
    plans: List[str]
    assignments: List[Tuple[str, str]]  # list of (eid, tool_call)


class ReWOOAgent:
    def __init__(
        self,
        llm,
        *,
        lab_test_mapping_df: Optional[pd.DataFrame] = None,
        include_ref_range: bool = False,
        bin_lab_results: bool = False,
        use_calculator: bool = False,
        ref_ranges: Optional[Dict[str, Any]] = None,
        calculator_critical_pct: float = 30.0,
        calculator_include_units: bool = True,
        debug_print_llm: bool = False,
    ):
        self.llm = llm
        self.lab_test_mapping_df = lab_test_mapping_df
        self.include_ref_range = include_ref_range
        self.bin_lab_results = bin_lab_results
        self.use_calculator = use_calculator
        self.ref_ranges = ref_ranges or {}
        self.calculator_critical_pct = calculator_critical_pct
        self.calculator_include_units = calculator_include_units
        self.debug_print_llm = bool(debug_print_llm)
        self._already_requested_scans: Dict[str, int] = {}

    def run_case(
        self,
        patient: Dict[str, Any],
        patient_history: str,
        patient_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        reset_patient_state()
        set_patient_history((patient_history or '').strip())
        self.patient = patient
        self._already_requested_scans = {}

        plan_res = self._plan(patient_history)
        evidences, resolved_inputs = self._execute(plan_res.assignments)
        worker_log = self._build_worker_log(plan_res.plans, plan_res.assignments, evidences, resolved_inputs)
        final_diag = self._solve(patient_history, worker_log)

        return {
            'planner': plan_res.raw,
            'worker_log': worker_log,
            'output': final_diag,
        }

    def _plan(self, patient_history: str) -> PlanResult:
        sys = _planner_system_prompt()
        usr = _planner_user_prompt(patient_history)
        raw = (self.llm([
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': usr},
        ], stop=[]) or '').strip()
        if self.debug_print_llm:
            try:
                thk, fin = _split_hermes_think(raw)
                print("[ReWOO Planner]\n" + (fin or raw))
            except Exception:
                pass
        plans, assignments = self._parse_planner_output(raw)
        return PlanResult(raw=raw, plans=plans, assignments=assignments)

    def _parse_planner_output(self, text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        plans: List[str] = []
        assigns: List[Tuple[str, str]] = []
        s = str(text or '')
        try:
            thk, fin = _split_hermes_think(s)
            if fin:
                s = fin
        except Exception:
            pass
        try:
            import re as _re
            s = _re.sub(r"<tool_call[^>]*>.*?</tool_call>", "\n", s, flags=_re.DOTALL | _re.IGNORECASE)
            s = s.replace('<|im_start|>', '\n').replace('<|im_end|>', '\n')
        except Exception:
            pass
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            m_plan = re.match(r'^plan\s*[:\-]\s*(.*)$', line, flags=re.I)
            if m_plan:
                plan_body = m_plan.group(1).strip()
                plans.append(f"Plan: {plan_body}" if plan_body else "Plan:")
                continue
            m = re.match(r'^#\s*e(\d+)\s*[:=]\s*(.+)$', line, flags=re.I)
            if m:
                eid = f"#E{m.group(1)}"
                tool_call = m.group(2).strip()
                assigns.append((eid, tool_call))
        return plans, assigns

    def _execute(self, assignments: List[Tuple[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
        evidences: Dict[str, str] = {}
        resolved_inputs: Dict[str, str] = {}
        for eid, tool_call in assignments:
            tool_name, arg = _extract_tool_and_input(tool_call)
            arg_text = str(arg or '')
            for var in re.findall(r'#E\d+', arg_text):
                if var in evidences:
                    arg_text = arg_text.replace(var, evidences[var])

            evidence = self._run_tool(tool_name, arg_text)
            evidences[eid] = evidence
            if arg is not None:
                resolved_inputs[eid] = arg_text
        return evidences, resolved_inputs

    def _build_worker_log(
        self,
        plans: List[str],
        assignments: List[Tuple[str, str]],
        evidences: Dict[str, str],
        resolved_inputs: Dict[str, str],
    ) -> str:
        tool_calls = {eid: tool_call for eid, tool_call in assignments}
        lines: List[str] = []
        if not plans:
            for eid, tool_call in assignments:
                lines.append("Plan: (no plan provided)")
                lines.append(f"{eid} = {_render_tool_call(tool_call, resolved_inputs.get(eid))}")
                lines.append("Evidence:")
                lines.append(str(evidences.get(eid, "No evidence found")).strip())
            return "\n".join(lines).strip()
        for idx, plan in enumerate(plans, 1):
            eid = f"#E{idx}"
            lines.append(plan)
            if eid in tool_calls:
                lines.append(f"{eid} = {_render_tool_call(tool_calls[eid], resolved_inputs.get(eid))}")
            lines.append("Evidence:")
            lines.append(str(evidences.get(eid, "No evidence found")).strip())
        return "\n".join(lines).strip()

    def _run_tool(self, tool_name: str, arg_text: str) -> str:
        name = _normalize_tool_name(tool_name)
        if name == 'Physical Examination':
            obs = physical_examination(self.patient)
            try:
                update_patient_observation('Physical Examination', None, obs)
            except Exception:
                pass
            return obs

        if name == 'Laboratory Tests':
            tests = _split_csv(arg_text)
            ai: List[Union[int, str]]
            if self.lab_test_mapping_df is not None:
                ai = convert_labs_to_itemid(tests, self.lab_test_mapping_df)
            else:
                ai = tests
            obs = lab_tests(
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
            try:
                update_patient_observation('Laboratory Tests', ai, obs)
            except Exception:
                pass
            return obs

        if name == 'Imaging':
            kv = _parse_keyvals(arg_text)
            try:
                if kv and ('region' in kv or 'modality' in kv):
                    ai = {'region': kv.get('region', ''), 'modality': kv.get('modality', '')}
                else:
                    ai = parse_imaging_action_input(arg_text)
            except Exception as e:
                err = f"Tool Error: Imaging input must contain a region and modality ({e})."
                try:
                    update_patient_observation('Imaging', arg_text, err)
                except Exception:
                    pass
                return err
            obs = imaging(self.patient, ai, already_requested_scans=self._already_requested_scans)
            try:
                update_patient_observation('Imaging', ai, obs)
            except Exception:
                pass
            return obs

        if name == 'ECG':
            obs = ecg(self.patient)
            try:
                update_patient_observation('ECG', None, obs)
            except Exception:
                pass
            return obs

        if name == 'Echocardiogram':
            obs = echocardiogram(self.patient)
            try:
                update_patient_observation('Echocardiogram', None, obs)
            except Exception:
                pass
            return obs

        return f"Tool Error: Unknown tool '{tool_name}'."

    def _solve(self, patient_history: str, worker_log: str) -> str:
        sys = _solver_system_prompt()
        usr = _solver_user_prompt(patient_history, worker_log)
        raw = (self.llm([
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': usr},
        ], stop=[]) or '').strip()
        return _normalize_final_diagnosis(raw)
