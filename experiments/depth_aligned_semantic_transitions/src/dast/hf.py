"""Lazy Hugging Face / PEFT construction for the real Rineya experiment."""

from __future__ import annotations

from typing import Any

from dast.config import DASTConfig


def select_precision(requested: str):
    import torch

    if requested == "fp32":
        return "fp32", torch.float32
    if requested == "bf16":
        return "bf16", torch.bfloat16
    if requested == "fp16":
        return "fp16", torch.float16
    if torch.cuda.is_available():
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return "bf16", torch.bfloat16
        return "fp16", torch.float16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "fp16", torch.float16
    return "fp32", torch.float32


def _quantization_config(config: DASTConfig):
    if config.model.quantization == "none":
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError("quantization requires transformers and bitsandbytes") from exc
    compute_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    if config.model.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_frozen_base(config: DASTConfig) -> tuple[Any, Any]:
    """Load the UNTOUCHED base model and tokenizer: no LoRA, no grads, eval mode.

    The frozen diagnostics (`dast.depth_cache`, `dast.depth_probe`) need the model
    as it was before any DAST training, because that is the reference point
    `FINDINGS_G3_SEEDS.md` section 2a compares against. Going through
    `build_hf_components` would attach a LoRA adapter whenever the run config
    enables one, so this path deliberately bypasses it rather than relying on
    zero-initialised `lora_B` being a mathematical no-op.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('real-model execution requires `pip install -e ".[train]"`') from exc

    model_cfg = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name_or_path,
        revision=model_cfg.revision,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _, dtype = select_precision(model_cfg.precision)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name_or_path,
        revision=model_cfg.revision,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=dtype,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def build_hf_components(config: DASTConfig, *, for_training: bool = True) -> tuple[Any, Any, dict[str, Any]]:
    """Load a causal LM and attach a matched LoRA adapter when configured."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('real-model execution requires `pip install -e ".[train]"`') from exc

    model_cfg = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name_or_path,
        revision=model_cfg.revision,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    precision_name, dtype = select_precision(model_cfg.precision)
    quantization = _quantization_config(config)
    kwargs: dict[str, Any] = {
        "revision": model_cfg.revision,
        "local_files_only": model_cfg.local_files_only,
        "trust_remote_code": model_cfg.trust_remote_code,
    }
    if quantization is not None:
        kwargs["quantization_config"] = quantization
    else:
        kwargs["torch_dtype"] = dtype
    if model_cfg.device_map:
        kwargs["device_map"] = model_cfg.device_map
    if model_cfg.attn_implementation:
        kwargs["attn_implementation"] = model_cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_cfg.name_or_path, **kwargs)
    info: dict[str, Any] = {
        "base_model": model_cfg.name_or_path,
        "revision": model_cfg.revision,
        "precision_requested": model_cfg.precision,
        "precision_resolved": precision_name,
        "quantization": model_cfg.quantization,
        "trust_remote_code": model_cfg.trust_remote_code,
        "local_files_only": model_cfg.local_files_only,
        "base_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }

    if quantization is not None and for_training:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("QLoRA requires peft") from exc
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
        info["kbit_prepared"] = True

    if for_training and config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
        info["gradient_checkpointing"] = True

    if config.lora.enabled:
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LoRA execution requires peft") from exc
        peft_config = PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=list(config.lora.target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        info["lora"] = {
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "target_modules": list(config.lora.target_modules),
        }
    elif for_training and config.training.freeze_base_without_lora:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        info["base_frozen_without_lora"] = True

    if for_training and config.training.fp32_trainable_params:
        promoted = 0
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
                parameter.data = parameter.data.float()
                promoted += 1
        info["fp32_trainable_tensors_promoted"] = promoted

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    info["trainable_parameter_count_before_dast_heads"] = trainable
    info["total_parameter_count"] = total
    info["trainable_fraction_before_dast_heads"] = trainable / total if total else 0.0
    return model, tokenizer, info
