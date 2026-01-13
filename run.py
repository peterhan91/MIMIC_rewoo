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
import logging
import os
import pickle
import random
from datetime import datetime
from os.path import join
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import transformers
from transformers import infer_device  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import torch

from rewoo_agent import ReWOOAgent
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
    enable_compile: bool = True,
    enable_static_kv: bool = True,
    attn_implementation: Optional[str] = None,  # e.g., "flash_attention_2", "eager"
    prompt_lookup_num_tokens: Optional[int] = None,
):
    """Return an LLM callable using transformers chat template.

    The returned callable accepts either a list of role messages or a plain string.
    Stop words are implemented via post-processing truncation.
    """
    # transformers and torch imported at module level

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    dtype = _to_torch_dtype(torch_dtype)
    quantization_config = None
    q = str(quant or "none").lower()
    if q not in {"none", "8bit", "4bit"}:
        q = "none"
    if q in {"8bit", "4bit"}:
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

    default_temperature = 0.2
    default_top_p = 1.0
    default_do_sample = False

    def call(
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        stop: Optional[List[str]] = None,
        do_sample: bool = default_do_sample,
        temperature: float = default_temperature,
        top_p: float = default_top_p,
    ) -> str:
        # Prepare inputs via chat template
        if isinstance(prompt_or_messages, list):
            messages = prompt_or_messages
        else:
            messages = [{"role": "user", "content": str(prompt_or_messages)}]

        input_ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        stopping = transformers.StoppingCriteriaList()
        if stop:
            stopping.append(_StopOnWords(input_ids.shape[-1], list(stop)))

        with torch.inference_mode():
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
        text = tok.decode(gen, skip_special_tokens=True)
        text = _truncate_on_stops(text, list(stop or []))
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


# (Removed OpenAI dependency; using local HF chat models.)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_hadm(pkl_path: str) -> Dict[str, Any]:
    return _load_pickle(pkl_path)


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
    p.add_argument("--include-ref-range", action="store_true")
    p.add_argument("--bin-lab-results", action="store_true")
    p.add_argument("--use-calculator", action="store_true")
    p.add_argument("--calculator-critical-pct", type=float, default=30.0)
    p.add_argument("--calculator-include-units", action="store_true")
    # Removed legacy final refinement/voting options (single-pass diagnosis)
    p.add_argument("--logdir", default="outputs_plain")
    p.add_argument("--run-descr", default="")
    p.add_argument("--first-patient", default=None, help="If set, only run this HADM id")
    p.add_argument("--max-patients", type=int, default=None, help="Maximum number of patients to process (for debugging)")
    p.add_argument("--sample-seed", type=int, default=42, help="Seed for sampling patients when --max-patients is set (deterministic subset)")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    p.add_argument("--debug-print-llm", action="store_true", help="Print raw assistant output each step")
    p.add_argument("--disable-torch-compile", action="store_true", help="Disable torch.compile wrapping (default: enabled)")
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
    requested_dtype = args.hf_dtype

    enable_compile = not args.disable_torch_compile
    enable_static_kv = not args.disable_static_kv
    attn_impl = None if str(args.attn_impl or "").lower() in {"", "auto"} else str(args.attn_impl)
    prompt_lookup_n = int(args.prompt_lookup_n) if int(args.prompt_lookup_n or 0) > 0 else None

    llm = _llm_hf_chat(
        args.hf_model_id,
        torch_dtype=requested_dtype,
        device_map=args.hf_device_map,
        max_new_tokens=args.max_new_tokens,
        quant=args.quant,
        enable_compile=enable_compile,
        enable_static_kv=enable_static_kv,
        attn_implementation=attn_impl,
        prompt_lookup_num_tokens=prompt_lookup_n,
    )

    # Load data
    hadm_info_clean = _load_hadm(args.hadm_pkl)
    lab_df: pd.DataFrame = _load_pickle(args.lab_map_pkl)
    # Auto-detect default ref ranges file if not provided
    if not args.ref_ranges_json:
        default_rr = join(os.getcwd(), "itemid_ref_ranges.json")
        if os.path.exists(default_rr):
            args.ref_ranges_json = default_rr
    ref_ranges = _load_ref_ranges(args.ref_ranges_json)
    

    # Run name
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(args.hadm_pkl))[0]
    model_tag = os.path.basename(args.hf_model_id).replace("/", "-")
    # Include agent + quantization tags in run name for clearer provenance
    agent_tag = "REWOO"
    quant_tag = str(args.quant or "none").replace(" ", "").lower()
    run_name = f"{base}_PLAIN_{model_tag}_AG{agent_tag}_Q{quant_tag}_{dt}{args.run_descr}"
    run_dir = join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    results_json = join(run_dir, f"{run_name}_results.json")

    # Debug options
    debug_print_llm = bool(args.debug or args.debug_print_llm)
    # ReWOO: Plan → Work → Solve with zero-shot diagnosis prompt
    agent = ReWOOAgent(
        llm=llm,
        lab_test_mapping_df=lab_df,
        include_ref_range=args.include_ref_range,
        bin_lab_results=args.bin_lab_results,
        use_calculator=args.use_calculator,
        ref_ranges=ref_ranges,
        calculator_critical_pct=args.calculator_critical_pct,
        calculator_include_units=args.calculator_include_units,
        debug_print_llm=debug_print_llm,
    )

    results: Dict[str, Any] = {}
    first_patient_seen = args.first_patient is None

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
