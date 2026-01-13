"""Minimal runner to perform automated agentic diagnosis on MIMIC-IV cases.

This mirrors MIMIC-ReAct/run.py behavior using the pure-Python agent and tools,
but uses local/transformers chat models (e.g., Llama 3 Instruct) instead of OpenAI.

Example:
  python run.py \
    --hadm-pkl /path/to/appendicitis_hadm_info_first_diag.pkl \
    --lab-map-pkl /path/to/lab_test_mapping.pkl \
    --ref-ranges-json /path/to/itemid_ref_ranges.json \
    --hf-model-id meta-llama/Meta-Llama-3-8B-Instruct

Notes:
  - The agent formats prompts as role-based messages for chat templates.
  - No LangChain or OpenAI dependency is required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from os.path import join
import pickle
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
import logging

import pandas as pd
import transformers
from transformers import infer_device  # type: ignore
from huggingface_hub import hf_hub_download
import random

from react_agent import ReActAgent, load_pickle
from rewoo_agent import ReWOOAgent

from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm.auto import tqdm  # type: ignore
import torch
torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

log = logging.getLogger(__name__)


def _to_torch_dtype(name: Optional[str]):
    if not name:
        return None
    n = str(name).lower()
    if n in {"auto", "none", ""}:
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(n, None)


def _infer_generation_device() -> str:
    if infer_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return str(infer_device())
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _empty_device_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("mps") and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()  # type: ignore[attr-defined]


def resilient_generate(model, *args, **kwargs):
    try:
        return model.generate(*args, **kwargs)
    except torch.OutOfMemoryError as exc:
        log.warning("Generation OOM; retrying with cache_implementation='offloaded'", exc_info=exc)
    device = _infer_generation_device()
    _empty_device_cache(device)
    retry_kwargs = dict(kwargs)
    retry_kwargs.setdefault("cache_implementation", "offloaded")
    return model.generate(*args, **retry_kwargs)


def _llm_hf_chat(
    model_id: str,
    *,
    torch_dtype: Optional[str] = "bfloat16",
    device_map: str = "balanced",
    max_new_tokens: int = 256,
    quant: str = "none",  # one of: none, 8bit, 4bit
    gpt_oss_reasoning_effort: Optional[str] = None,
    enable_compile: bool = True,
    enable_flash_attention: bool = True,
    hermes_thinking: Optional[bool] = None,
    hermes_keep_cots: Optional[bool] = None,
    enable_static_kv: bool = True,
    attn_implementation: Optional[str] = None,  # e.g., "flash_attention_2", "eager"
    prompt_lookup_num_tokens: Optional[int] = None,
):
    """Return an LLM callable using transformers chat template.

    The returned callable accepts either a list of role messages or a plain string.
    Stop words are implemented via post-processing truncation.
    """
    # transformers and torch imported at module level

    model_id_lower = model_id.lower()
    tokenizer_kwargs: Dict[str, Any] = {}
    # Detect model families that need special handling
    is_hermes4 = ("hermes-4" in model_id_lower) or ("nousresearch/hermes-4" in model_id_lower)
    is_qwen_thinking = ("qwen3-next" in model_id_lower) and ("thinking" in model_id_lower)
    is_phi_reasoning = ("phi-4-reasoning-plus" in model_id_lower) or ("microsoft/phi-4-reasoning-plus" in model_id_lower)
    if "magistral" in model_id_lower:
        tokenizer_kwargs.update({"tokenizer_type": "mistral", "use_fast": False})
    if is_hermes4:
        # Hermes 4 exposes custom chat template knobs via remote code
        tokenizer_kwargs.update({"trust_remote_code": True})
    if is_qwen_thinking:
        # Qwen3-Next-80B-A3B-Thinking provides a custom template; keep remote code if needed
        tokenizer_kwargs.update({"trust_remote_code": True})

    tok = transformers.AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    is_gpt_oss = model_id_lower.startswith("openai/gpt-oss")
    dtype = _to_torch_dtype(torch_dtype)
    quantization_config = None
    q = str(quant or "none").lower()
    if q not in {"none", "8bit", "4bit"}:
        q = "none"
    skip_quant = is_gpt_oss or str(torch_dtype or "").lower() in {"mxfp4"}
    if q in {"8bit", "4bit"} and not skip_quant:
        if q == "8bit":
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        elif q == "4bit":
            quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)

    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    # Optional attention implementation preference (e.g., flash_attention_2)
    if attn_implementation and str(attn_implementation).lower() not in {"", "auto"}:
        try:
            model_kwargs["attn_implementation"] = str(attn_implementation)
        except Exception:
            pass
    if is_gpt_oss:
        model_kwargs["trust_remote_code"] = True
        model_kwargs.pop("quantization_config", None)
    if is_hermes4:
        model_kwargs["trust_remote_code"] = True
    if is_qwen_thinking:
        model_kwargs["trust_remote_code"] = True

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
    except TypeError:
        # Retry without attention implementation if unsupported by this transformers version/model
        if "attn_implementation" in model_kwargs:
            mk2 = dict(model_kwargs)
            mk2.pop("attn_implementation", None)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **mk2)
        else:
            raise

    # If set, try to update attention implementation dynamically as well
    if attn_implementation and str(attn_implementation).lower() not in {"", "auto"}:
        try:
            model.set_attention_implementation(str(attn_implementation))  # type: ignore[attr-defined]
        except Exception:
            pass

    # Enable static kv-cache if requested (helps pair with forward compile)
    if enable_static_kv:
        try:
            model.generation_config.cache_implementation = "static"
        except Exception:
            pass

    # Compile: prefer compiling forward with static kv; else whole-model compile
    if enable_compile and hasattr(torch, "compile"):
        # Allow graph breaks and dynamic shapes; fall back to eager on errors
        try:
            import torch._dynamo as dynamo  # type: ignore
            dynamo.config.suppress_errors = True
        except Exception:
            pass
        try:
            kw = dict(mode="reduce-overhead", fullgraph=False, dynamic=True)
            if enable_static_kv:
                model.forward = torch.compile(model.forward, **kw)  # type: ignore[assignment]
            else:
                model = torch.compile(model, **kw)  # type: ignore[assignment]
        except TypeError:
            # Older PyTorch: retry without dynamic=
            try:
                kw = dict(mode="reduce-overhead", fullgraph=False)
                if enable_static_kv:
                    model.forward = torch.compile(model.forward, **kw)  # type: ignore[assignment]
                else:
                    model = torch.compile(model, **kw)  # type: ignore[assignment]
            except Exception:
                pass
        except Exception:
            pass

    # Build default terminators (EOS + end-of-turn if available)
    eos_token_id = tok.eos_token_id
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    term_ids = [i for i in [eos_token_id, eot_id] if isinstance(i, int) and i >= 0]

    class _StopOnWords(transformers.StoppingCriteria):
        def __init__(self, start_len: int, stop_words: List[str]):
            super().__init__()
            self.start_len = start_len
            self.stop_words = stop_words or []

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
            if not self.stop_words:
                return False
            gen_ids = input_ids[0][self.start_len:]
            if gen_ids.numel() == 0:
                return False
            text = tok.decode(gen_ids, skip_special_tokens=True)
            return any(sw in text for sw in self.stop_words)

    def _truncate_on_stops(text: str, stops: Optional[List[str]]) -> str:
        if not text or not stops:
            return text
        idxs = [text.find(sw) for sw in stops if sw]
        idxs = [i for i in idxs if i >= 0]
        if not idxs:
            return text
        cut = min(idxs)
        return text[:cut].rstrip()

    use_magistral_defaults = "magistral" in model_id_lower
    use_hermes_defaults = bool(is_hermes4)
    use_qwen_defaults = bool(is_qwen_thinking)
    use_phi_defaults = bool(is_phi_reasoning)
    default_temperature = (
        0.7 if use_magistral_defaults else (
        0.6 if (use_hermes_defaults or use_qwen_defaults) else (
        0.8 if use_phi_defaults else 0.2))
    )
    default_top_p = 0.95 if (use_magistral_defaults or use_hermes_defaults or use_qwen_defaults or use_phi_defaults) else 1.0
    default_do_sample = bool(use_magistral_defaults or use_hermes_defaults or use_qwen_defaults or use_phi_defaults)

    def _extract_gpt_oss_final_message(raw: str) -> str:
        """Return only the final-channel content from GPT-OSS outputs."""
        if not raw:
            return raw

        final_marker = "<|channel|>final<|message|>"
        idx = raw.rfind(final_marker)
        if idx != -1:
            tail = raw[idx + len(final_marker):]
            end_idx = tail.find("<|end|>")
            if end_idx != -1:
                tail = tail[:end_idx]
            tail = tail.strip()
            if tail:
                return tail

        # If we didn't find explicit channel markers, fall back to trimming
        # everything before the last assistant delimiter.
        start_marker = "<|start|>assistant"
        if start_marker in raw:
            tail = raw.split(start_marker)[-1]
            end_idx = tail.find("<|end|>")
            if end_idx != -1:
                tail = tail[:end_idx]
            tail = tail.strip()
            if tail:
                # GPT-OSS usually prefixes the channel marker here; remove if present.
                if tail.startswith("<|channel|>final<|message|>"):
                    tail = tail[len("<|channel|>final<|message|>"):].strip()
                return tail

        # Final fallback: tokens may collapse to "assistantfinal" without
        # explicit channel markers once special tokens are stripped. Grab the
        # substring after the last such marker.
        matches = list(re.finditer(r"assistantfinal", raw))
        if matches:
            tail = raw[matches[-1].end():].strip()
            if tail:
                return tail

        return raw

    def call(
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        stop: Optional[List[str]] = None,
        do_sample: bool = default_do_sample,
        temperature: float = default_temperature,
        top_p: float = default_top_p,
        tools: Optional[List[Any]] = None,
    ) -> str:
        # Prepare inputs via chat template
        if isinstance(prompt_or_messages, list):
            messages = prompt_or_messages
        else:
            messages = [{"role": "user", "content": str(prompt_or_messages)}]

        # For Hermes 4, optionally strengthen the instruction to actually emit <think>…</think>
        if is_hermes4 and hermes_thinking:
            try:
                hint = (
                    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
                )
                # Find system message if present
                sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
                if sys_idx is None:
                    messages = ([{"role": "system", "content": hint}] + messages)
                else:
                    sys_msg = messages[sys_idx]
                    content = sys_msg.get("content")
                    if isinstance(content, str):
                        if "</think>" not in content and "<think>" not in content:
                            joiner = "\n\n" if content and not content.endswith(("\n", " ")) else ""
                            messages[sys_idx]["content"] = content + joiner + hint
                    elif isinstance(content, list):
                        messages[sys_idx]["content"] = list(content) + [{"type": "text", "text": hint}]
                    else:
                        messages[sys_idx]["content"] = hint
            except Exception:
                pass

        extra_kwargs: Dict[str, Any] = {}
        if tools is not None:
            extra_kwargs["tools"] = tools
        if is_gpt_oss:
            extra_kwargs["model_identity"] = "You are OpenAI GPT OSS."
            if gpt_oss_reasoning_effort:
                extra_kwargs["reasoning_effort"] = str(gpt_oss_reasoning_effort).lower()
        # Hermes 4: optionally enable hybrid reasoning (<think>…</think>)
        if is_hermes4 and hermes_thinking is True:
            extra_kwargs["thinking"] = True
        if is_hermes4 and hermes_keep_cots is True:
            # Keep the content between <think>...</think> in outputs
            extra_kwargs["keep_cots"] = True
        # Qwen3-Next-Thinking always runs with thinking mode via its template; no flag needed

        try:
            input_ids = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                **extra_kwargs,
            ).to(model.device)
        except TypeError:
            fallback_kwargs_list: List[Dict[str, Any]] = []
            if extra_kwargs:
                # Remove optional fields that some templates may not accept
                fallback_kwargs_list.append({k: v for k, v in extra_kwargs.items() if k not in {"model_identity", "reasoning_effort", "thinking", "keep_cots"}})
                # Try with only tools (common supported arg)
                if "tools" in extra_kwargs:
                    fallback_kwargs_list.append({"tools": extra_kwargs["tools"]})
            fallback_kwargs_list.append({})
            input_ids = None
            for kwargs in fallback_kwargs_list:
                try:
                    input_ids = tok.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        **kwargs,
                    ).to(model.device)
                    break
                except TypeError:
                    continue
            if input_ids is None:
                input_ids = tok.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)

        stopping = transformers.StoppingCriteriaList()
        if stop:
            stopping.append(_StopOnWords(input_ids.shape[-1], list(stop)))

        sdpa_context = nullcontext()
        # Use SDPA FlashAttention kernel unless user selected a non-FA attention impl
        wants_fa = enable_flash_attention and (
            (attn_implementation is None)
            or (str(attn_implementation).lower() in {"", "auto", "flash_attention_2", "flash_attention_3"})
        )
        if wants_fa and SDPBackend and sdpa_kernel:
            try:
                sdpa_context = nullcontext()
            except Exception:
                sdpa_context = nullcontext()

        with torch.inference_mode():
            with sdpa_context:
                gen_kwargs: Dict[str, Any] = dict(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    eos_token_id=term_ids if term_ids else None,
                    stopping_criteria=stopping,
                )
                # Phi-4-reasoning-plus recommended sampling includes top_k
                if use_phi_defaults:
                    try:
                        gen_kwargs["top_k"] = 50
                    except Exception:
                        pass
                # Prompt lookup decoding support (greedy/sampling only)
                if isinstance(prompt_lookup_num_tokens, int) and prompt_lookup_num_tokens > 0:
                    gen_kwargs["prompt_lookup_num_tokens"] = int(prompt_lookup_num_tokens)
                try:
                    outputs = resilient_generate(model, **gen_kwargs)
                except TypeError:
                    # Retry without prompt_lookup_num_tokens if unsupported
                    if "prompt_lookup_num_tokens" in gen_kwargs:
                        g2 = dict(gen_kwargs)
                        g2.pop("prompt_lookup_num_tokens", None)
                        outputs = resilient_generate(model, **g2)
                    else:
                        raise
        gen = outputs[0][input_ids.shape[-1]:]
        # Keep special tokens when needed for parsing:
        # - Hermes 4: preserve <think> when keep_cots, and preserve <tool_call> tags when tools are used
        # - Qwen Thinking: preserve special tokens
        preserve_special_for_tools = bool(is_hermes4 and tools is not None)
        skip_special = not ((is_hermes4 and (hermes_keep_cots or preserve_special_for_tools)) or is_qwen_thinking)
        text = tok.decode(gen, skip_special_tokens=skip_special)
        text = _truncate_on_stops(text, list(stop or []))
        if is_gpt_oss:
            text = _extract_gpt_oss_final_message(text)
        return text

    return call


def _load_ref_ranges(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _maybe_enable_rmm_unified_memory(managed_memory: bool = True) -> bool:
    """Enable RAPIDS RMM as the CUDA allocator for PyTorch (GH200 unified memory).

    Returns True if successfully enabled; otherwise returns False silently.
    """
    try:
        if not torch.cuda.is_available():
            return False
        import rmm  # type: ignore
        from rmm.allocators.torch import rmm_torch_allocator  # type: ignore

        # Initialize RMM with managed (unified) memory so GPU can oversubscribe to host RAM
        rmm.reinitialize(managed_memory=bool(managed_memory))
        # Instruct PyTorch to use the RMM allocator for all CUDA allocations
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        return True
    except Exception:
        return False


def _load_magistral_system_prompt(model_id: str) -> Optional[Dict[str, Any]]:
    """Load the structured Magistral system prompt with optional [THINK] block."""
    try:
        file_path = hf_hub_download(repo_id=model_id, filename="SYSTEM_PROMPT.txt")
    except Exception:
        return None

    try:
        with open(file_path, "r") as f:
            system_prompt = f.read()
    except Exception:
        return None

    think_start = system_prompt.find("[THINK]")
    think_end = system_prompt.find("[/THINK]")
    if 0 <= think_start < think_end:
        return {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt[:think_start]},
                {
                    "type": "thinking",
                    "thinking": system_prompt[think_start + len("[THINK]") : think_end],
                    "closed": True,
                },
                {"type": "text", "text": system_prompt[think_end + len("[/THINK]") :]},
            ],
        }

    return {"role": "system", "content": system_prompt}


# (Removed OpenAI dependency; using local HF chat models.)


def _load_hadm(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description="Pure-Python MIMIC automated diagnosis runner (HF models)")
    p.add_argument("--hadm-pkl", required=True, help="Pickle file with HADM info dict (e.g., appendicitis_hadm_info_first_diag.pkl)")
    p.add_argument("--lab-map-pkl", required=True, help="Pickle to lab test mapping DataFrame (from MIMIC-ReAct)")
    p.add_argument("--ref-ranges-json", default=None, help="Optional JSON with itemid ref ranges")
    p.add_argument("--hf-model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Hugging Face model id (chat/instruct)")
    p.add_argument("--hf-dtype", default="bfloat16", help="Torch dtype: auto|bfloat16|float16|float32")
    p.add_argument("--hf-device-map", default="balanced", help="transformers device_map (e.g., 'auto')")
    p.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens to generate per call")
    p.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"], help="Quantization mode")
    # Hermes 4 optional reasoning blocks
    p.add_argument("--hermes-thinking", action="store_true", help="Enable Hermes <think> reasoning blocks in chat template (Hermes 4 only)")
    p.add_argument("--hermes-keep-cots", action="store_true", help="Keep <think> content in outputs (Hermes 4 only)")
    p.add_argument("--llama-ipython-mode", dest="llama_ipython_mode", action="store_true", default=None, help="Emit tool outputs as ipython messages (<|python_tag|>...<|eot_id|>) for Llama 3.1 custom tool calling")
    p.add_argument("--no-llama-ipython-mode", dest="llama_ipython_mode", action="store_false", default=None, help="Disable ipython tool output mode")
    # Include function definitions block in prompt (useful for Llama 3.2 lightweight zero-shot function calling)
    p.add_argument("--include-fn-defs-in-prompt", dest="include_fn_defs", action="store_true", default=None, help="Include function definitions JSON block in system prompt")
    p.add_argument("--no-include-fn-defs-in-prompt", dest="include_fn_defs", action="store_false", default=None, help="Do not include function definitions block in prompt")
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--include-ref-range", action="store_true")
    p.add_argument("--bin-lab-results", action="store_true")
    p.add_argument("--use-calculator", action="store_true")
    p.add_argument("--calculator-critical-pct", type=float, default=30.0)
    p.add_argument("--calculator-include-units", action="store_true")
    p.add_argument("--provide-diagnostic-criteria", action="store_true")
    # Removed legacy final refinement/voting options (single-pass diagnosis)
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium", help="Optional reasoning effort hint (mirrors GPT-OSS reasoning levels)")
    p.add_argument("--logdir", default="outputs_plain")
    p.add_argument("--run-descr", default="")
    p.add_argument("--first-patient", default=None, help="If set, only run this HADM id")
    p.add_argument("--max-patients", type=int, default=None, help="Maximum number of patients to process (for debugging)")
    p.add_argument("--sample-seed", type=int, default=42, help="Seed for sampling patients when --max-patients is set (deterministic subset)")
    # Agent type: iterative ReAct (default) vs. ReWOO (plan-then-execute)
    p.add_argument("--agent-type", choices=["react", "rewoo"], default="react", help="Agent algorithm: ReAct (iterative) or ReWOO (plan-then-execute)")
    p.add_argument("--state-gather", action="store_true", help="Only collect patient state snapshots; skip Final Diagnosis and downstream recommendations.")
    # ReWOO planner reflexion (optional)
    p.add_argument("--rewoo-planner-reflexion", action="store_true", help="Enable planner reflexion memory for ReWOO (disables labs-only reflexion)")
    p.add_argument("--rewoo-planner-reflexion-mem", type=int, default=3, help="Max reflections to inject into planner prompts")
    # ReWOO multi-plan selection
    p.add_argument("--rewoo-num-plans", type=int, default=1, help="Number of planner candidates to sample and score (select best)")
    p.add_argument("--rewoo-plan-temperature", type=float, default=0.9, help="Sampling temperature for planner when num_plans>1")
    p.add_argument("--rewoo-plan-top-p", type=float, default=0.98, help="Top-p for planner when num_plans>1")
    # Final Diagnosis enrichment
    p.add_argument("--final-include-ddx", action="store_true", help="Include diagnosis artifacts (diagnosis, expert DDx, expert reasoning) in Final Diagnosis input")
    # Final Diagnosis private reasoning hint (enabled by default; can disable)
    default_reasoning_hint = (
        "Think step by step privately; do not reveal your chain of thought. "
        "Use only the Patient State (JSON). When present, consider ‘diagnosis’ (top-10) and ‘expert DDx’ as guidance, "
        "but base decisions strictly on the evidence. If evidence is insufficient, return ‘Insufficient data — gather more observations first.’ "
        "Output exactly one JSON object following the schema; keep ‘rationale’ concise and evidence-grounded; include a diverse set of high-likelihood ‘differential_diagnosis’ items and a calibrated ‘confidence’ in [0,1]."
    )
    p.add_argument("--final-reasoning-hint", default=default_reasoning_hint, help="Private reasoning hint appended to Final Diagnosis system prompt")
    p.add_argument("--no-final-reasoning-hint", dest="final_reasoning_hint", action="store_const", const="", help="Disable extra reasoning hint for Final Diagnosis")
    # Tool-calling defaults to ON; allow disabling via --no-tool-calling
    p.add_argument("--tool-calling", dest="tool_calling", action="store_true", default=True, help="Enable HF tool-calling mode (default: on)")
    p.add_argument("--no-tool-calling", dest="tool_calling", action="store_false", help="Disable HF tool-calling mode")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug (print + save messages)")
    p.add_argument("--debug-print-messages", action="store_true", help="Print chat messages each step")
    p.add_argument("--debug-print-llm", action="store_true", help="Print raw assistant output each step")
    p.add_argument("--debug-save-messages", action="store_true", help="Save chat messages to files under the run dir")
    p.add_argument("--disable-torch-compile", action="store_true", help="Disable torch.compile wrapping (default: enabled)")
    p.add_argument("--disable-flash-attn", action="store_true", help="Disable Flash Attention SDPA kernel (default: enabled)")
    # New optimization knobs
    p.add_argument("--disable-static-kv", action="store_true", help="Disable static kv-cache optimization (default: enabled)")
    p.add_argument("--attn-impl", default="flash_attention_2", help="Attention implementation preference for model (e.g., 'flash_attention_2', 'eager', or 'auto')")
    p.add_argument("--prompt-lookup-n", type=int, default=0, help="Enable prompt lookup decoding with N overlap tokens (0 disables)")
    # GH200 / RAPIDS RMM unified memory (optional)
    p.add_argument("--enable-rmm", action="store_true", help="Use RAPIDS RMM as PyTorch CUDA allocator (GH200 unified memory)")
    p.add_argument("--rmm-managed-memory", dest="rmm_managed_memory", action="store_true", default=True, help="Enable RMM managed (unified) memory when --enable-rmm is set (default: on)")
    p.add_argument("--no-rmm-managed-memory", dest="rmm_managed_memory", action="store_false", help="Disable RMM managed (unified) memory when --enable-rmm is set")
    args = p.parse_args()

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    #     handlers=[logging.StreamHandler()],
    # )

    if tqdm is None:
        def log(message: str) -> None:
            print(message)

        log("tqdm not installed; falling back to standard console output.")
    else:
        def log(message: str) -> None:
            tqdm.write(str(message))

    os.makedirs(args.logdir, exist_ok=True)

    # Optionally enable RAPIDS RMM unified memory allocator before loading models
    if getattr(args, "enable_rmm", False):
        ok = _maybe_enable_rmm_unified_memory(managed_memory=bool(getattr(args, "rmm_managed_memory", True)))
        if ok:
            tqdm.write("RMM unified memory allocator enabled for PyTorch CUDA.") if tqdm else print("RMM unified memory allocator enabled for PyTorch CUDA.")
        else:
            tqdm.write("RMM not available or failed to initialize; continuing with default allocator.") if tqdm else print("RMM not available or failed to initialize; continuing with default allocator.")

    # Prepare HF transformers LLM callable
    model_id_lower = args.hf_model_id.lower()
    requested_dtype = args.hf_dtype
    if model_id_lower.startswith("openai/gpt-oss") and requested_dtype in {None, "", "bfloat16", "bf16"}:
        requested_dtype = "auto"

    gpt_oss_reasoning_effort = args.reasoning_effort if model_id_lower.startswith("openai/gpt-oss") else None

    enable_compile = not args.disable_torch_compile
    enable_flash_attn = not args.disable_flash_attn
    enable_static_kv = not args.disable_static_kv
    attn_impl = None if str(args.attn_impl or "").lower() in {"", "auto"} else str(args.attn_impl)
    prompt_lookup_n = int(args.prompt_lookup_n) if int(args.prompt_lookup_n or 0) > 0 else None

    llm = _llm_hf_chat(
        args.hf_model_id,
        torch_dtype=requested_dtype,
        device_map=args.hf_device_map,
        max_new_tokens=args.max_new_tokens,
        quant=args.quant,
        gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
        enable_compile=enable_compile,
        enable_flash_attention=enable_flash_attn,
        enable_static_kv=enable_static_kv,
        attn_implementation=attn_impl,
        prompt_lookup_num_tokens=prompt_lookup_n,
        hermes_thinking=bool(getattr(args, "hermes_thinking", False)),
        hermes_keep_cots=bool(getattr(args, "hermes_keep_cots", False)),
    )
    final_llm = llm
    # Auto-enable ipython mode for Llama 3.1 if not explicitly set
    llama_ipython_mode = args.llama_ipython_mode
    include_fn_defs = args.include_fn_defs
    if llama_ipython_mode is None or include_fn_defs is None:
        mid = str(args.hf_model_id or "").lower()
        is_llama31 = ("3.1" in mid)
        is_llama32 = ("3.2" in mid)
        is_llama33 = ("3.3" in mid)
        if llama_ipython_mode is None:
            # Enable ipython mode only for Llama 3.1 by default
            llama_ipython_mode = bool(is_llama31)
        if include_fn_defs is None:
            # Include function definitions block for Llama 3.2/3.3 lightweight models by default
            include_fn_defs = bool(is_llama32 or is_llama33)

    # Load data
    hadm_info_clean = _load_hadm(args.hadm_pkl)
    lab_df: pd.DataFrame = load_pickle(args.lab_map_pkl)
    # Auto-detect default ref ranges file if not provided
    if not args.ref_ranges_json:
        default_rr = join(os.getcwd(), "itemid_ref_ranges.json")
        if os.path.exists(default_rr):
            args.ref_ranges_json = default_rr
    ref_ranges = _load_ref_ranges(args.ref_ranges_json)
    

    # Run name
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(args.hadm_pkl))[0]
    disease_name = base
    for suffix in ("_hadm_info_first_diag", "_hadm_info"):
        if disease_name.endswith(suffix):
            disease_name = disease_name[: -len(suffix)]
            break
    disease_name = disease_name or base
    model_tag = os.path.basename(args.hf_model_id).replace("/", "-")
    # Include agent + quantization tags in run name for clearer provenance
    agent_tag = str(args.agent_type or "react").upper()
    quant_tag = str(args.quant or "none").replace(" ", "").lower()
    run_name = f"{base}_PLAIN_{model_tag}_AG{agent_tag}_Q{quant_tag}_{dt}{args.run_descr}"
    run_dir = join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    results_json = join(run_dir, f"{run_name}_results.json")
    patient_states_pkl = join(run_dir, f"{model_tag}_{disease_name}.pkl")

    # Debug options
    debug_print_messages = bool(args.debug or args.debug_print_messages)
    debug_print_llm = bool(args.debug or args.debug_print_llm)
    debug_save_messages = bool(args.debug or args.debug_save_messages)
    debug_dir = None
    if debug_save_messages:
        debug_dir = join(run_dir, "messages")

    system_prompt_override = None
    if "magistral" in model_id_lower:
        system_prompt_override = _load_magistral_system_prompt(args.hf_model_id)
        if system_prompt_override is None:
            log("Magistral system prompt could not be loaded; falling back to default instructions.")

    if args.agent_type == "react":
        agent = ReActAgent(
            llm=llm,
            final_llm=final_llm,
            lab_test_mapping_df=lab_df,
            include_ref_range=args.include_ref_range,
            bin_lab_results=args.bin_lab_results,
            use_calculator=args.use_calculator,
            ref_ranges=ref_ranges,
            calculator_critical_pct=args.calculator_critical_pct,
            calculator_include_units=args.calculator_include_units,
            provide_diagnostic_criteria=args.provide_diagnostic_criteria,
            max_iterations=args.max_iters,
            tool_calling=bool(args.tool_calling),
            llama_ipython_mode=bool(llama_ipython_mode),
            include_function_defs_in_prompt=bool(include_fn_defs),
            system_prompt_override=system_prompt_override,
            debug_print_messages=debug_print_messages,
            debug_print_llm=debug_print_llm,
            debug_save_messages=debug_save_messages,
            debug_dir=debug_dir,
            patient_state_pkl_path=patient_states_pkl,
            state_gather_only=bool(args.state_gather),
            include_ddx_in_final=bool(args.final_include_ddx),
            final_reasoning_hint=str(args.final_reasoning_hint or ""),
        )
    else:
        # ReWOO: Plan → Work → Solve with Final Diagnosis tool
        agent = ReWOOAgent(
            llm=llm,
            lab_test_mapping_df=lab_df,
            provide_diagnostic_criteria=args.provide_diagnostic_criteria,
            include_ref_range=args.include_ref_range,
            bin_lab_results=args.bin_lab_results,
            use_calculator=args.use_calculator,
            ref_ranges=ref_ranges,
            calculator_critical_pct=args.calculator_critical_pct,
            calculator_include_units=args.calculator_include_units,
            debug_print_messages=debug_print_messages,
            debug_print_llm=debug_print_llm,
            planner_reflexion=bool(args.rewoo_planner_reflexion),
            planner_reflexion_mem_max=int(args.rewoo_planner_reflexion_mem),
            planner_num_candidates=max(1, int(args.rewoo_num_plans or 1)),
            planner_temperature=float(args.rewoo_plan_temperature or 0.7),
            planner_top_p=float(args.rewoo_plan_top_p or 0.9),
            patient_state_pkl_path=patient_states_pkl,
            state_gather_only=bool(args.state_gather),
            include_ddx_in_final=bool(args.final_include_ddx),
            final_reasoning_hint=str(args.final_reasoning_hint or ""),
        )

    results: Dict[str, Any] = {}
    first_patient_seen = args.first_patient is None
    patient_count = 0

    # Sample patients if max_patients is set
    patient_items = list(hadm_info_clean.items())
    # Make the base order stable across runs by sorting on HADM id
    try:
        patient_items.sort(key=lambda kv: str(kv[0]))
    except Exception:
        pass
    if args.max_patients and args.max_patients > 0:
        rnd = random.Random(int(args.sample_seed))
        patient_items = rnd.sample(patient_items, min(args.max_patients, len(patient_items)))

    total_patients = len(patient_items)
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            patient_items,
            total=total_patients,
            desc="Processing patients",
            unit="case",
            leave=True,
        )
        iterator = enumerate(progress_bar, 1)
    else:
        iterator = enumerate(patient_items, 1)

    for idx, (hadm_id, patient) in iterator:
        if not first_patient_seen:
            if str(hadm_id) == str(args.first_patient):
                first_patient_seen = True
            else:
                continue

        patient_count += 1
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"HADM {hadm_id}")
        else:
            log(f"Processing case {idx}/{total_patients}: HADM {hadm_id}")

        history = (patient.get("Patient History", "") or "").strip()
        out = agent.run_case(patient, history, patient_id=hadm_id)
        # Ensure JSON-serializable keys (pickle may yield numpy.int64 keys)
        results[str(hadm_id)] = out

        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if progress_bar is not None:
        progress_bar.close()

    # Save final results
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"Completed processing {len(results)} patients")
    log(f"Final results saved to {results_json}")


if __name__ == "__main__":
    main()
