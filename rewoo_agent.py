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
     updates the Patient State with each observation.
  3) Solver: calls the existing final_diagnosis() tool using only Patient
     State to maximize diagnostic accuracy, with the option to pass a
     concise evidence log as notes.

Focus: maximize diagnostic accuracy while maintaining efficiency.
  - One LLM call for planning (vs. iterative ReAct)
  - Finalization via a single-pass diagnosis generator

Usage: wired into run.py via --agent-type rewoo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import re
import os
import pickle

import pandas as pd

from tools import (
    physical_examination,
    lab_tests,
    imaging,
    ecg,
    echocardiogram,
    diagnostic_criteria,
    patient_state_tool,
    final_diagnosis,
    medication_recommendation,
    generate_differential_diagnosis,
    get_last_cot,
    reset_patient_state,
    set_patient_history,
    update_patient_observation,
)
from tools import procedure_recommendation as _proc

# Reuse robust helpers from the ReAct agent
from react_agent import (
    convert_labs_to_itemid,
    parse_imaging_action_input,
)


# ----------------------------- Planner prompt -----------------------------

def _tool_descriptions(include_diag_criteria: bool) -> str:
    lines: List[str] = []
    lines.append("Tools (choose minimally; prefer high-yield steps):")
    lines.append("Physical Examination[input]: Perform physical examination; input can be left blank.")
    lines.append("Laboratory Tests[input]: Order specific tests, provide a comma-separated list. Choose targeted tests for the suspected organ system(s).")
    lines.append("Imaging[input]: Do specific imaging; input must specify region and modality.")
    # Add explicit indications to encourage case-relevant cardiology usage
    lines.append(
        "ECG[input]: Retrieve the ECG report; only ordering when cardiac symptoms/signs are present."
    )
    lines.append(
        "Echocardiogram[input]: Retrieve the echocardiogram (Echo) report; only ordering when clinically indicated."
    )
    lines.append("Patient State[input]: Read or update patient state; use 'get' or JSON.")
    lines.append("Procedure Recommendation[input]: Generate procedures from state + final diagnosis; returns JSON with queries and predicted ICD9/10 codes.")
    lines.append("Medication Recommendation[input]: Review Medications on Admission and propose per-medication actions; returns JSON mapping medication -> {action, dose}.")
    if include_diag_criteria:
        lines.append("Diagnostic Criteria[input]: Examine diagnostic criteria for a pathology; input is a single condition name.")
    lines.append("\nImaging Regions: Abdomen, Chest, Head, Neck, Pelvis")
    lines.append("Imaging Modalities: Ultrasound, CT, MRI, Radiograph")
    return "\n".join(lines)


def _planner_system_prompt(include_diag_criteria: bool, reflections_block: Optional[str] = None) -> str:
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
        + ("" if not reflections_block else (reflections_block + "\n\n"))
        + _tool_descriptions(include_diag_criteria)
    )


def _planner_user_prompt(patient_history: str) -> str:
    return (
        "Patient History:\n" + (patient_history or "") + "\n\n"
        "Begin! Output only Plan/#E lines as specified. Keep it concise and tailored.\n"
    )


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
        provide_diagnostic_criteria: bool = False,
        include_ref_range: bool = False,
        bin_lab_results: bool = False,
        use_calculator: bool = False,
        ref_ranges: Optional[Dict[str, Any]] = None,
        calculator_critical_pct: float = 30.0,
        calculator_include_units: bool = True,
        debug_print_messages: bool = False,
        debug_print_llm: bool = False,
        # Planner reflexion controls
        planner_reflexion: bool = False,
        planner_reflexion_mem_max: int = 3,
        # Multi-plan sampling/selection
        planner_num_candidates: int = 1,
        planner_temperature: float = 0.9,
        planner_top_p: float = 0.98,
        patient_state_pkl_path: Optional[str] = None,
        state_gather_only: bool = False,
        include_ddx_in_final: bool = False,
        final_reasoning_hint: str = "",
    ):
        self.llm = llm
        self.lab_test_mapping_df = lab_test_mapping_df
        self.provide_diagnostic_criteria = bool(provide_diagnostic_criteria)
        self.include_ref_range = include_ref_range
        self.bin_lab_results = bin_lab_results
        self.use_calculator = use_calculator
        self.ref_ranges = ref_ranges or {}
        self.calculator_critical_pct = calculator_critical_pct
        self.calculator_include_units = calculator_include_units
        self.debug_print_messages = bool(debug_print_messages)
        self.debug_print_llm = bool(debug_print_llm)
        self._already_requested_scans: Dict[str, int] = {}
        self._imaging_unavailable: bool = False
        self._ecg_unavailable: bool = False
        # Planner reflexion memory
        self.planner_reflexion = bool(planner_reflexion)
        self._planner_reflexion_mem_max = int(planner_reflexion_mem_max or 0)
        self._planner_reflections: List[str] = []
        # Multi-plan
        self._planner_num_candidates = max(1, int(planner_num_candidates or 1))
        self._planner_temperature = float(planner_temperature)
        self._planner_top_p = float(planner_top_p)
        self._last_planner_candidates: List[Dict[str, Any]] = []
        # Post-solver medication bookkeeping
        self._med_rec_done: bool = False
        self._last_med_rec: Optional[str] = None
        # Track availability of a confirmed final diagnosis
        self._final_diagnosis_ready: bool = False
        self._patient_state_pkl_path = patient_state_pkl_path
        self._patient_state_cache: Dict[str, Any] = {}
        self._current_patient_id: Optional[Union[str, int]] = None
        self._state_gather_only = bool(state_gather_only)
        self._include_ddx_in_final = bool(include_ddx_in_final)
        self._final_reasoning_hint = str(final_reasoning_hint or "")
        if self._patient_state_pkl_path and os.path.exists(self._patient_state_pkl_path):
            try:
                with open(self._patient_state_pkl_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    self._patient_state_cache = loaded
            except Exception:
                self._patient_state_cache = {}

    def _add_ddx_fields(self, container: Dict[str, Any], ddx_text: str) -> None:
        try:
            obj = json.loads(ddx_text)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            try:
                if "diagnosis" in obj:
                    container["diagnosis"] = obj.get("diagnosis")
                if "expert DDx" in obj:
                    container["expert DDx"] = obj.get("expert DDx")
                if "expert reasoning" in obj:
                    container["expert reasoning"] = obj.get("expert reasoning")
            except Exception:
                pass

    # ----------------------------- Orchestration -----------------------------

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

    def run_case(self, patient: Dict[str, Any], patient_history: str, patient_id: Optional[Union[str, int]] = None) -> Dict[str, Any]:
        reset_patient_state()
        set_patient_history((patient_history or "").strip())
        self.patient = patient
        # Reset imaging repeat gating per case
        self._already_requested_scans = {}
        self._imaging_unavailable = False
        self._ecg_unavailable = False
        self._med_rec_done = False
        self._last_med_rec = None
        self._final_diagnosis_ready = False
        # Reset planner reflections per case
        if self.planner_reflexion:
            self._planner_reflections = []
        self._current_patient_id = patient_id

        # 1) Plan
        plan_res = self._plan(patient_history)

        # 2) Work
        evidences, worker_log = self._execute(plan_res.assignments)

        # 2.5) Labs-only reflexion audit (always-on): ensure lab bundle covers key questions
        lab_reflex = self._lab_reflex_replan(patient_history, plan_res.raw, worker_log)
        worker_log_lab_reflex = ""
        if lab_reflex.assignments:
            _, worker_log_lab_reflex = self._execute(lab_reflex.assignments)

        if self._state_gather_only:
            # Run diagnosis step to enrich state before snapshot (optional in state-gather mode)
            try:
                ddx_out = generate_differential_diagnosis(self.llm, notes="")
                ddx_cot = get_last_cot("differential_diagnosis_top10").get("thinking", "")
                try:
                    update_patient_observation("Diagnosis", None, ddx_out)
                except Exception:
                    pass
            except Exception:
                ddx_out = "Tool Error during diagnosis"
                ddx_cot = ""
            self._persist_patient_state()
            try:
                state_snapshot = patient_state_tool("get")
            except Exception:
                state_snapshot = json.dumps({"state_gather_mode": True, "error": "Unable to serialize patient state"}, ensure_ascii=False)
            result_state = {
                "planner": plan_res.raw,
                "planner_cot": _split_hermes_think(plan_res.raw)[0],
                "worker_log": worker_log,
                "lab_reflex_planner": lab_reflex.raw,
                "lab_reflex_planner_cot": _split_hermes_think(lab_reflex.raw)[0],
                "worker_log_lab_reflex": worker_log_lab_reflex,
                "differential_diagnosis": ddx_out,
                "differential_diagnosis_cot": ddx_cot,
                "output": state_snapshot,
                "medication_recommendation": "Skipped: state gather mode",
                "procedure_recommendation": "Skipped: state gather mode",
                "replanned": False,
                "state_gather_mode": True,
            }
            self._add_ddx_fields(result_state, ddx_out)
            if self.planner_reflexion:
                result_state["planner_reflections"] = list(self._planner_reflections)
            self._current_patient_id = None
            return result_state

        # 3) Solve (Final Diagnosis tool). If a re-plan is clearly needed (e.g., imaging unavailable),
        #    avoid firing Final Diagnosis before re-planning to save an unnecessary LLM call.
        combined_log_for_replan = (worker_log + ("\n\n" + worker_log_lab_reflex if worker_log_lab_reflex else "")).strip()
        if self._imaging_unavailable or self._ecg_unavailable:
            # Prime reflections before re-planning, so they influence the re-plan prompt
            if self.planner_reflexion:
                self._maybe_store_planner_reflection(plan_res.raw, combined_log_for_replan, "")
            repl = self._replan(patient_history, plan_res.raw, combined_log_for_replan)
            evid2, worker_log2 = self._execute(repl.assignments)
            self._persist_patient_state()
            # Compute diagnoses just before solving
            try:
                ddx_out2 = generate_differential_diagnosis(self.llm, notes="")
                ddx_cot2 = get_last_cot("differential_diagnosis_top10").get("thinking", "")
                try:
                    update_patient_observation("Diagnosis", None, ddx_out2)
                except Exception:
                    pass
            except Exception:
                ddx_out2 = "Tool Error during diagnosis"
                ddx_cot2 = ""
            out_after_replan = final_diagnosis(
                self.llm,
                notes="",
                include_ddx=self._include_ddx_in_final,
                reasoning_hint=self._final_reasoning_hint,
            )
            self._mark_final_diagnosis_ready(out_after_replan)
            # Order fields to show lab reflex info before final output
            result_obj = {
                "planner": plan_res.raw,
                "planner_cot": _split_hermes_think(plan_res.raw)[0],
                "worker_log": worker_log,
                "lab_reflex_planner": lab_reflex.raw,
                "lab_reflex_planner_cot": _split_hermes_think(lab_reflex.raw)[0],
                "worker_log_lab_reflex": worker_log_lab_reflex,
                "replanned": True,
                "planner_2": repl.raw,
                "planner_2_cot": _split_hermes_think(repl.raw)[0],
                "worker_log_2": worker_log2,
                "differential_diagnosis": ddx_out2,
                "differential_diagnosis_cot": ddx_cot2,
                "output": out_after_replan,
            }
            self._add_ddx_fields(result_obj, ddx_out2)
            # Update planner reflections memory after a weak attempt (for next runs)
            if self.planner_reflexion:
                self._maybe_store_planner_reflection(plan_res.raw, (worker_log + "\n\n" + worker_log_lab_reflex).strip(), out_after_replan)
                result_obj["planner_reflections"] = list(self._planner_reflections)
            if self._final_diagnosis_ready:
                # Ensure Medication Recommendation post-solver
                try:
                    med_out = self._last_med_rec or medication_recommendation(self.llm, self.patient, notes="")
                    result_obj["medication_recommendation"] = med_out
                    # Log CoT
                    result_obj["final_diagnosis_cot"] = get_last_cot("final_diagnosis").get("thinking", "")
                    result_obj["medication_recommendation_cot"] = get_last_cot("medication_recommendation").get("thinking", "")
                    if not self._med_rec_done:
                        try:
                            update_patient_observation("Medication Recommendation", None, med_out)
                        except Exception:
                            pass
                except Exception:
                    result_obj["medication_recommendation"] = "Tool Error during medication recommendation"
                # Procedure Recommendation post-solver
                try:
                    fd = self._extract_final_diagnosis_text(out_after_replan)
                    proc_out = _proc(self.llm, self.patient, final_diagnosis=fd)
                    result_obj["procedure_recommendation"] = proc_out
                    result_obj["procedure_recommendation_cot"] = get_last_cot("procedure_recommendation").get("thinking", "")
                except Exception:
                    result_obj["procedure_recommendation"] = "Tool Error during procedure recommendation"
            else:
                result_obj["medication_recommendation"] = "Skipped: final diagnosis not available."
                result_obj["procedure_recommendation"] = "Skipped: final diagnosis not available."
            self._current_patient_id = None
            return result_obj

        # Otherwise, proceed to final diagnosis now
        self._persist_patient_state()
        # Compute diagnoses just before solving
        try:
            ddx_out = generate_differential_diagnosis(self.llm, notes="")
            ddx_cot = get_last_cot("differential_diagnosis_top10").get("thinking", "")
            try:
                update_patient_observation("Diagnosis", None, ddx_out)
            except Exception:
                pass
        except Exception:
            ddx_out = "Tool Error during diagnosis"
            ddx_cot = ""
        out = final_diagnosis(
            self.llm,
            notes="",
            include_ddx=self._include_ddx_in_final,
            reasoning_hint=self._final_reasoning_hint,
        )
        self._mark_final_diagnosis_ready(out)
        # Order fields so lab reflex info appears before the final output
        result: Dict[str, Any] = {
            "planner": plan_res.raw,
            "planner_cot": _split_hermes_think(plan_res.raw)[0],
            "worker_log": worker_log,
            "lab_reflex_planner": lab_reflex.raw,
            "lab_reflex_planner_cot": _split_hermes_think(lab_reflex.raw)[0],
            "worker_log_lab_reflex": worker_log_lab_reflex,
            "differential_diagnosis": ddx_out,
            "differential_diagnosis_cot": ddx_cot,
            "output": out,
        }
        self._add_ddx_fields(result, ddx_out)
        if self._final_diagnosis_ready:
            try:
                med_out = self._last_med_rec or medication_recommendation(self.llm, self.patient, notes="")
                result["medication_recommendation"] = med_out
                result["final_diagnosis_cot"] = get_last_cot("final_diagnosis").get("thinking", "")
                result["medication_recommendation_cot"] = get_last_cot("medication_recommendation").get("thinking", "")
                if not self._med_rec_done:
                    try:
                        update_patient_observation("Medication Recommendation", None, med_out)
                    except Exception:
                        pass
            except Exception:
                result["medication_recommendation"] = "Tool Error during medication recommendation"
            try:
                fd = self._extract_final_diagnosis_text(out)
                proc_out = _proc(self.llm, self.patient, final_diagnosis=fd)
                result["procedure_recommendation"] = proc_out
                result["procedure_recommendation_cot"] = get_last_cot("procedure_recommendation").get("thinking", "")
            except Exception:
                result["procedure_recommendation"] = "Tool Error during procedure recommendation"
        else:
            result["medication_recommendation"] = "Skipped: final diagnosis not available."
            result["procedure_recommendation"] = "Skipped: final diagnosis not available."

        # Fallback: re-plan once if model explicitly signals insufficient data
        if self._is_insufficient(out):
            # Prime reflections before insufficient-data re-plan
            if self.planner_reflexion:
                self._maybe_store_planner_reflection(plan_res.raw, combined_log_for_replan, out)
            repl = self._replan(patient_history, plan_res.raw, combined_log_for_replan)
            evid2, worker_log2 = self._execute(repl.assignments)
            self._persist_patient_state()
            # Refresh diagnoses on updated evidence before re-solve
            try:
                ddx_out = generate_differential_diagnosis(self.llm, notes="")
                ddx_cot = get_last_cot("differential_diagnosis_top10").get("thinking", "")
                try:
                    update_patient_observation("Diagnosis", None, ddx_out)
                except Exception:
                    pass
            except Exception:
                ddx_out = "Tool Error during diagnosis"
                ddx_cot = ""
            out2 = final_diagnosis(
                self.llm,
                notes="",
                include_ddx=self._include_ddx_in_final,
                reasoning_hint=self._final_reasoning_hint,
            )
            self._mark_final_diagnosis_ready(out2)
            result.update({
                "replanned": True,
                "planner_2": repl.raw,
                "planner_2_cot": _split_hermes_think(repl.raw)[0],
                "worker_log_2": worker_log2,
                "differential_diagnosis": ddx_out,
                "differential_diagnosis_cot": ddx_cot,
                "output": out2,
            })
            self._add_ddx_fields(result, ddx_out)
            if self._final_diagnosis_ready:
                try:
                    med_out2 = self._last_med_rec or medication_recommendation(self.llm, self.patient, notes="")
                    result["medication_recommendation"] = med_out2
                    result["final_diagnosis_cot"] = get_last_cot("final_diagnosis").get("thinking", "")
                    result["medication_recommendation_cot"] = get_last_cot("medication_recommendation").get("thinking", "")
                    if not self._med_rec_done:
                        try:
                            update_patient_observation("Medication Recommendation", None, med_out2)
                        except Exception:
                            pass
                except Exception:
                    result["medication_recommendation"] = "Tool Error during medication recommendation"
                try:
                    fd2 = self._extract_final_diagnosis_text(out2)
                    proc_out2 = _proc(self.llm, self.patient, final_diagnosis=fd2)
                    result["procedure_recommendation"] = proc_out2
                    result["procedure_recommendation_cot"] = get_last_cot("procedure_recommendation").get("thinking", "")
                except Exception:
                    result["procedure_recommendation"] = "Tool Error during procedure recommendation"
            else:
                result["medication_recommendation"] = "Skipped: final diagnosis not available."
                result["procedure_recommendation"] = "Skipped: final diagnosis not available."
        else:
            result["replanned"] = False
        # Update planner reflections memory at end (for next runs)
        if self.planner_reflexion:
            final_out_text = str(result.get("output") or "")
            if self._is_insufficient(final_out_text):
                self._maybe_store_planner_reflection(
                    plan_res.raw,
                    (worker_log + ("\n\n" + worker_log_lab_reflex if worker_log_lab_reflex else "")).strip(),
                    final_out_text,
                )
            result["planner_reflections"] = list(self._planner_reflections)
        self._current_patient_id = None
        return result

    # ----------------------------- Planner -----------------------------

    def _plan(self, patient_history: str) -> PlanResult:
        sys = _planner_system_prompt(
            include_diag_criteria=self.provide_diagnostic_criteria,
            reflections_block=self._reflections_block(),
        )
        usr = _planner_user_prompt(patient_history)
        self._last_planner_candidates = []
        n = self._planner_num_candidates
        if n <= 1:
            raw = (self.llm([
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ], stop=[]) or "").strip()
            if self.debug_print_llm:
                try:
                    thk, fin = _split_hermes_think(raw)
                    print("[ReWOO Planner]\n" + (fin or raw))
                except Exception:
                    pass
            plans, assignments = self._parse_planner_output(raw)
            return PlanResult(raw=raw, plans=plans, assignments=assignments)
        # Sample multiple candidates and score
        best_idx = 0
        best_score = -1.0
        best_raw = ""
        for i in range(n):
            temp = min(1.0, max(0.0, self._planner_temperature + 0.1 * i))
            raw_i = (self.llm([
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ], stop=[], do_sample=True, temperature=temp, top_p=self._planner_top_p) or "").strip()
            score_i = self._score_plan(raw_i, patient_history)
            self._last_planner_candidates.append({"plan": raw_i, "score": score_i, "temperature": temp})
            if score_i > best_score:
                best_score = score_i
                best_idx = i
                best_raw = raw_i
        if self.debug_print_llm:
            try:
                print("[ReWOO Planner Candidates]" )
                for j, c in enumerate(self._last_planner_candidates):
                    thk, fin = _split_hermes_think(c['plan'])
                    body = fin if thk else c['plan']
                    print(f"-- Candidate {j} (score={c['score']:.3f}, T={c['temperature']:.2f})\n{body}")
            except Exception:
                pass
        plans, assignments = self._parse_planner_output(best_raw)
        return PlanResult(raw=best_raw, plans=plans, assignments=assignments)

    def _score_plan(self, raw_plan: str, patient_history: str) -> float:
        """LLM-based rubric scoring of a plan, returns 0..1."""
        try:
            sys = (
                "You are a strict grader of medical evidence-gathering plans. "
                "Score how well the plan is efficient and sufficient for a safe final diagnosis given the tools. "
                "Criteria: (1) covers Physical Examination early; (2) orders essential labs when relevant and targeted to the suspected organ system(s)—avoid generic all-panels without justification; "
                "(3) uses a single best imaging choice with explicit region/modality only if indicated; (4) uses cardiology tools appropriately and only with indications; "
                "(5) minimizes steps without losing safety; (6) avoids duplicating identical steps; (7) correct tool syntax; (8) addresses given Reflections. "
                "Penalize default checklists without clear indications, multiple initial imaging steps, or steps unlikely to change management. Output a single number 0..1 with up to 3 decimals."
            )
            usr = (
                (self._reflections_block() + "\n\n" if self._reflections_block() else "")
                + "Patient History:\n" + (patient_history or "") + "\n\n"
                + "Plan to score:\n" + (raw_plan or "") + "\n"
            )
            resp = (self.llm([
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ], stop=[], do_sample=False, temperature=0.0, top_p=1.0) or "").strip()
            m = re.search(r"\d+(?:\.\d+)?", resp)
            val = float(m.group(0)) if m else 0.0
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0

    def _parse_planner_output(self, text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        plans: List[str] = []
        assigns: List[Tuple[str, str]] = []
        s = str(text or "")
        # Be robust to Hermes/Qwen special formatting: strip think blocks and tool_call tags
        try:
            thk, fin = _split_hermes_think(s)
            if fin:
                s = fin
        except Exception:
            pass
        try:
            import re as _re
            s = _re.sub(r"<tool_call[^>]*>.*?</tool_call>", "\n", s, flags=_re.DOTALL | _re.IGNORECASE)
            s = s.replace("<|im_start|>", "\n").replace("<|im_end|>", "\n")
        except Exception:
            pass
        for line in s.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("plan:"):
                plans.append(s)
                continue
            # Match lines like '#E1 = Tool[input]' or '#E1=Tool[input]'
            m = re.match(r"^#E(\d+)\s*=\s*(.+)$", s)
            if m:
                eid = f"#E{m.group(1)}"
                tool_call = m.group(2).strip()
                assigns.append((eid, tool_call))
        return plans, assigns

    # ----------------------------- Worker -----------------------------

    def _execute(self, assignments: List[Tuple[str, str]]) -> Tuple[Dict[str, str], str]:
        evidences: Dict[str, str] = {}
        logs: List[str] = []
        step_idx = 0
        for eid, tool_call in assignments:
            step_idx += 1
            # Resolve #E references within the raw input content (if any)
            tool_name, arg = _extract_tool_and_input(tool_call)
            arg_text = str(arg or "")
            for var in re.findall(r"#E\d+", arg_text):
                if var in evidences:
                    arg_text = arg_text.replace(var, evidences[var])

            evidence = self._run_tool(tool_name, arg_text)
            evidences[eid] = evidence
            logs.append(f"{eid} = {tool_name}[{arg if arg is not None else ''}]\n{evidence}\n")
        worker_log = "\n".join(logs).strip()
        return evidences, worker_log

    def _run_tool(self, tool_name: str, arg_text: str) -> str:
        name = str(tool_name or "").strip()
        # Physical Examination
        if name.lower().startswith("physical examination"):
            obs = physical_examination(self.patient)
            try:
                update_patient_observation("Physical Examination", None, obs)
            except Exception:
                pass
            return obs

        # Patient State
        if name.lower().startswith("patient state"):
            ai: Union[str, Dict[str, Any]] = arg_text or "get"
            if ai.strip().lower() in {"", "[]", "{}", "none"}:
                ai = "get"
            # Accept raw JSON
            if isinstance(ai, str) and ai.strip().startswith("{"):
                try:
                    ai = json.loads(ai)
                except Exception:
                    pass
            obs = patient_state_tool(ai)
            try:
                update_patient_observation("Patient State", ai, obs)
            except Exception:
                pass
            return obs

        # Laboratory Tests
        if name.lower().startswith("laboratory tests"):
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
                update_patient_observation("Laboratory Tests", ai, obs)
            except Exception:
                pass
            return obs

        # Imaging
        if name.lower().startswith("imaging"):
            kv = _parse_keyvals(arg_text)
            ai: Dict[str, str]
            try:
                if kv and ("region" in kv or "modality" in kv):
                    ai = {"region": kv.get("region", ""), "modality": kv.get("modality", "")}
                else:
                    ai = parse_imaging_action_input(arg_text)
            except Exception as e:
                err = f"Tool Error: Imaging input must contain a region and modality ({e})."
                try:
                    update_patient_observation("Imaging", arg_text, err)
                except Exception:
                    pass
                return err
            # Track repeat gating across the plan
            obs = imaging(self.patient, ai, already_requested_scans=self._already_requested_scans)
            # Track imaging unavailability for re-plan trigger
            if "not available" in str(obs or "").lower():
                self._imaging_unavailable = True
            try:
                update_patient_observation("Imaging", ai, obs)
            except Exception:
                pass
            return obs

        # ECG
        if name.lower().startswith("ecg"):
            obs = ecg(self.patient)
            if "not available" in str(obs or "").lower():
                self._ecg_unavailable = True
            try:
                update_patient_observation("ECG", None, obs)
            except Exception:
                pass
            return obs

        # Echocardiogram
        if name.lower().startswith("echocardiogram"):
            obs = echocardiogram(self.patient)
            try:
                update_patient_observation("Echocardiogram", None, obs)
            except Exception:
                pass
            return obs

        # Diagnostic Criteria
        if name.lower().startswith("diagnostic criteria"):
            topic = (arg_text or "").strip()
            obs = diagnostic_criteria(topic)
            try:
                update_patient_observation("Diagnostic Criteria", topic, obs)
            except Exception:
                pass
            return obs

        # Procedure Recommendation
        if name.lower().startswith("procedure"):
            if not self._final_diagnosis_ready:
                return "Tool Error: Final diagnosis must be completed before Procedure Recommendation."
            fd = (arg_text or "").strip()
            if not fd:
                return "Tool Error: Procedure Recommendation requires final_diagnosis as input."
            try:
                # accept JSON object containing final_diagnosis too
                if fd.startswith("{"):
                    obj = json.loads(fd)
                    if isinstance(obj, dict):
                        fd2 = str(obj.get("final_diagnosis", "")).strip()
                        if fd2:
                            fd = fd2
            except Exception:
                pass
            obs = _proc(self.llm, self.patient, final_diagnosis=fd)
            try:
                update_patient_observation("Procedure Recommendation", fd, obs)
            except Exception:
                pass
            return obs

        # Medication Recommendation
        if name.lower().startswith("medication"):
            if not self._final_diagnosis_ready:
                return "Tool Error: Final diagnosis must be completed before Medication Recommendation."
            notes = (arg_text or "").strip()
            obs = medication_recommendation(self.llm, self.patient, notes=notes)
            try:
                update_patient_observation("Medication Recommendation", notes, obs)
            except Exception:
                pass
            # Track for post-solver reuse
            self._med_rec_done = True
            self._last_med_rec = obs
            return obs

        # Unknown tool → return an informative error observation
        return f"Tool Error: Unknown tool '{tool_name}'."

    # ----------------------------- Re-plan once -----------------------------

    def _is_insufficient(self, output: str) -> bool:
        s = str(output or "").lower()
        return "insufficient data" in s

    def _extract_final_diagnosis_text(self, output: str) -> str:
        try:
            obj = json.loads(output)
            if isinstance(obj, dict):
                fd = obj.get("final_diagnosis")
                if isinstance(fd, str) and fd.strip():
                    return fd.strip()
        except Exception:
            pass
        m = re.search(r'"final_diagnosis"\s*:\s*"([^"]+)"', str(output or ""))
        return m.group(1).strip() if m else ""

    def _mark_final_diagnosis_ready(self, output: str) -> None:
        if self._final_diagnosis_ready:
            return
        fd = self._extract_final_diagnosis_text(output)
        if fd:
            self._final_diagnosis_ready = True

    def _replan(self, patient_history: str, prior_plan_raw: str, worker_log: str) -> PlanResult:
        """Request at most 2 additional high-yield steps to close information gaps.

        Avoid referencing #E variables from the prior plan; rely on current Patient State instead.
        """
        guidance_ecg = ""
        if self._ecg_unavailable:
            guidance_ecg = "If ECG was unavailable, consider ordering an Echocardiogram if clinically indicated.\n\n"
        sys = (
            "The previous attempt concluded with insufficient data for a safe final diagnosis. "
            "Propose AT MOST TWO additional, high-yield steps to close the most critical information gaps. "
            "Avoid duplicating identical steps already executed unless strictly necessary. "
            "Prefer complementary labs or an alternative imaging modality if a prior imaging was unavailable.\n\n"
            + guidance_ecg
            + (self._reflections_block() + "\n\n" if self._reflections_block() else "")
            + "Return ONLY lines in the same schema as before:\n"
            + "Plan: <concise rationale>\n"
            + "#E1 = <Tool>[<Input>]\n"
            + "Plan: <next step>\n"
            + "#E2 = <Tool>[<Input>]\n\n"
            + _tool_descriptions(self.provide_diagnostic_criteria)
            + "\n\nDo NOT reference #E variables from earlier plans; base your choices on the current Patient State."
        )
        # Provide compact context: summary of prior plan and evidence snippets
        usr = (
            "Patient History:\n" + (patient_history or "") + "\n\n"
            "Executed Plan (read-only):\n" + (prior_plan_raw or "").strip() + "\n\n"
            "Observed Evidence (read-only):\n" + (worker_log or "").strip() + "\n\n"
            "Now propose at most two additional steps. Output only Plan/#E lines."
        )
        raw = (self.llm([
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ], stop=[]) or "").strip()
        if self.debug_print_llm:
            try:
                thk, fin = _split_hermes_think(raw)
                print("[ReWOO Re-Plan]\n" + (fin or raw))
            except Exception:
                pass
        plans, assignments = self._parse_planner_output(raw)
        return PlanResult(raw=raw, plans=plans, assignments=assignments)

    def _lab_reflex_replan(self, patient_history: str, prior_plan_raw: str, worker_log: str) -> PlanResult:
        """Always-offered, labs-only reflexion step to bundle complementary tests.

        Guidance: "Reflexion: Double-check that your lab order covers the key questions (inflammation, infection, organ function) and consider bundling complementary tests instead of single isolated labs."

        Returns at most one additional Laboratory Tests step (or none if satisfied).
        """
        guidance = (
            "Reflexion: Double-check that your lab order covers the key questions (inflammation, infection, organ function) "
            "and consider bundling complementary tests instead of single isolated labs.\n\n"
        )
        sys = (
            "You are auditing the current evidence with a focus on LABS ONLY. "
            "If the current lab order does not adequately cover inflammation, infection, and core organ function, propose ONE bundled lab order. "
            "If coverage is already sufficient, output nothing.\n\n"
            + guidance
            + (self._reflections_block() + "\n\n" if self._reflections_block() else "")
            + "Return ONLY lines using the schema and restrict to Laboratory Tests:\n"
            "Plan: <concise rationale>\n"
            "#E1 = Laboratory Tests[<comma-separated tests>]\n\n"
        )
        usr = (
            "Patient History:\n" + (patient_history or "") + "\n\n"
            "Executed Plan (read-only):\n" + (prior_plan_raw or "").strip() + "\n\n"
            "Observed Evidence (read-only):\n" + (worker_log or "").strip() + "\n\n"
            "Now output at most one labs bundle step only if needed."
        )
        raw = (self.llm([
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ], stop=[]) or "").strip()
        if self.debug_print_llm:
            try:
                thk, fin = _split_hermes_think(raw)
                print("[ReWOO Lab Reflexion]\n" + (fin or raw))
            except Exception:
                pass
        plans, assignments = self._parse_planner_output(raw)
        # Filter to labs-only just in case
        labs_only = []
        for eid, tool_call in assignments:
            tool_name, _ = _extract_tool_and_input(tool_call)
            if str(tool_name or "").strip().lower().startswith("laboratory tests"):
                labs_only.append((eid, tool_call))
        # Keep at most one labs step
        if len(labs_only) > 1:
            labs_only = [labs_only[0]]
        return PlanResult(raw=raw, plans=plans, assignments=labs_only)

    # ----------------------------- Planner Reflexion helpers -----------------------------

    def _reflections_block(self) -> str:
        if not self.planner_reflexion:
            return ""
        if not self._planner_reflections:
            return ""
        lines = ["Reflections from prior attempts (read-only):"]
        for r in self._planner_reflections[-self._planner_reflexion_mem_max:]:
            r_line = str(r).strip()
            if not r_line:
                continue
            if not r_line.startswith("-"):
                r_line = "- " + r_line
            lines.append(r_line)
        return "\n".join(lines)

    def _maybe_store_planner_reflection(
        self,
        prior_plan_raw: str,
        worker_log: str,
        final_output: str,
    ) -> None:
        # Trigger only when enabled
        if not self.planner_reflexion:
            return
        try:
            refl = self._generate_planner_reflection(prior_plan_raw, worker_log, final_output)
        except Exception:
            refl = ""
        r = str(refl or "").strip()
        if not r:
            return
        # Avoid duplicate reflections when the same guidance is generated in multiple phases
        if r in self._planner_reflections:
            return
        self._planner_reflections.append(r)
        # Cap memory
        if self._planner_reflexion_mem_max > 0 and len(self._planner_reflections) > self._planner_reflexion_mem_max:
            self._planner_reflections = self._planner_reflections[-self._planner_reflexion_mem_max:]

    def _generate_planner_reflection(
        self,
        prior_plan_raw: str,
        worker_log: str,
        final_output: str,
    ) -> str:
        """Generate a concise, actionable plan-improvement for the next attempt."""
        sys = (
            "You will see a prior planning attempt, the observed evidence, and the final output. "
            "Write a brief plan-improvement for the next attempt with 3-6 concrete bullets. "
            "Focus on: high-yield labs to add/bundle, better imaging region/modality choices, patient-state updates, and critical diagnostic criteria to check. "
            "Do not restate the case; provide actionable steps referencing exact tool names and inputs. "
            "Return only bullet lines starting with '- '."
        )
        usr = (
            "Prior Plan (read-only):\n" + (prior_plan_raw or "") + "\n\n"
            + "Observed Evidence (read-only):\n" + (worker_log or "") + "\n\n"
            + ("Final Output (read-only):\n" + (final_output or "") + "\n\n" if final_output else "")
            + ("Prior Reflections (read-only):\n" + "\n".join(self._planner_reflections) + "\n\n" if self._planner_reflections else "")
            + "Now output only the improved plan bullets."
        )
        raw = (self.llm([
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ], stop=[]) or "").strip()
        # Keep only bullet lines
        bullets = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("- "):
                bullets.append(s)
        return "\n".join(bullets) if bullets else raw
