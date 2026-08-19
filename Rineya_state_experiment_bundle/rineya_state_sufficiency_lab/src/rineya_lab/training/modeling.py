"""Base model / tokenizer / LoRA / QLoRA construction.

Everything here is lazy: importing this module does not import torch. The base
checkpoint is loaded identically for both arms of the experiment; the only thing
a condition changes is the data view it is trained on.
"""

from __future__ import annotations

from typing import Any

from rineya_lab.config import LabConfig, ModelConfig
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("training.modeling")


def select_precision(requested: str) -> tuple[str, Any]:
    """Resolve ``auto`` to bf16 / fp16 / fp32 based on real hardware support.

    bf16 is preferred whenever the device supports it: it removes the loss-scale
    tuning that fp16 needs, which is one less uncontrolled difference between
    two runs that are supposed to differ only in training data.
    """
    import torch

    if requested == "fp32":
        return "fp32", torch.float32
    if requested == "bf16":
        return "bf16", torch.bfloat16
    if requested == "fp16":
        return "fp16", torch.float16

    if torch.cuda.is_available():
        supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if supported:
            return "bf16", torch.bfloat16
        return "fp16", torch.float16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple Silicon has unified memory: an fp32 base can exhaust the host and
        # push the process into swap, where step times vary by two orders of
        # magnitude and destroy any timing comparability between arms. fp16 halves
        # the resident base; trainable LoRA parameters are kept in fp32 separately.
        return "fp16", torch.float16
    return "fp32", torch.float32


def build_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    from rineya_lab.training.chat import ensure_pad_token

    name = cfg.tokenizer_name_or_path or cfg.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
        use_fast=True,
    )
    pad_info = ensure_pad_token(tokenizer)
    LOGGER.info("Loaded tokenizer %s (pad handling: %s)", name, pad_info)
    return tokenizer


def _quantization_config(cfg: ModelConfig):
    if cfg.quantization == "none":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.bnb_4bit_compute_dtype
    ]
    if cfg.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def build_model(cfg: LabConfig, *, for_training: bool = True) -> tuple[Any, dict[str, Any]]:
    """Load the base model and (optionally) attach a LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM

    model_cfg = cfg.model
    precision_name, dtype = select_precision(model_cfg.precision)
    quant_config = _quantization_config(model_cfg)

    kwargs: dict[str, Any] = {
        "revision": model_cfg.revision,
        "trust_remote_code": model_cfg.trust_remote_code,
        "local_files_only": model_cfg.local_files_only,
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["dtype"] = dtype
    if model_cfg.device_map:
        kwargs["device_map"] = model_cfg.device_map
    if model_cfg.attn_implementation:
        kwargs["attn_implementation"] = model_cfg.attn_implementation

    LOGGER.info("Loading base model %s (precision=%s quantization=%s)", model_cfg.name_or_path, precision_name, model_cfg.quantization)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name_or_path, **kwargs)
    except TypeError:
        # transformers < 4.56 uses torch_dtype rather than dtype
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name_or_path, **kwargs)

    info: dict[str, Any] = {
        "precision_requested": model_cfg.precision,
        "precision_resolved": precision_name,
        "quantization": model_cfg.quantization,
        "base_model": model_cfg.name_or_path,
        "revision": model_cfg.revision,
        "base_parameter_count": sum(p.numel() for p in model.parameters()),
    }

    if quant_config is not None and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
        )
        info["kbit_prepared"] = True

    if for_training and cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
        info["gradient_checkpointing"] = True

    if cfg.lora.enabled:
        model, lora_info = attach_lora(model, cfg)
        info.update(lora_info)
    else:
        info["lora"] = "disabled"

    if for_training and cfg.training.fp32_trainable_params and precision_name in ("fp16", "bf16"):
        promoted = cast_trainable_params_to_fp32(model)
        info["fp32_trainable_params_promoted"] = promoted

    if not for_training:
        model.eval()
    device_hint = "cuda" if torch.cuda.is_available() else "cpu"
    info["device_hint"] = model_cfg.device_map or device_hint
    return model, info


def attach_lora(model, cfg: LabConfig) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    lora_cfg = cfg.lora
    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=list(lora_cfg.target_modules),
        modules_to_save=list(lora_cfg.modules_to_save) or None,
        task_type=lora_cfg.task_type,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("LoRA attached: %d trainable / %d total parameters (%.4f%%)", trainable, total, 100 * trainable / total)
    return model, {
        "lora": {
            "r": lora_cfg.r,
            "alpha": lora_cfg.alpha,
            "dropout": lora_cfg.dropout,
            "bias": lora_cfg.bias,
            "target_modules": list(lora_cfg.target_modules),
            "modules_to_save": list(lora_cfg.modules_to_save),
            "task_type": lora_cfg.task_type,
        },
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def cast_trainable_params_to_fp32(model) -> int:
    """Promote trainable (LoRA) parameters to fp32 while the base stays low precision.

    Standard mixed-precision recipe. Without it, small adapter updates vanish into
    fp16 rounding, and the two arms of the comparison could differ for numerical
    reasons unrelated to the training representation.
    """
    import torch

    promoted = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.float()
            promoted += 1
    if promoted:
        LOGGER.info("Promoted %d trainable tensors to fp32 (frozen base stays low precision)", promoted)
    return promoted


def load_model_for_inference(cfg: LabConfig, adapter_path: str | None = None):
    """Load the base model plus an optional trained adapter for evaluation."""
    model, info = build_model(cfg, for_training=False)
    if adapter_path:
        from peft import PeftModel

        base_had_adapter = cfg.lora.enabled
        if base_had_adapter:
            # build_model already wrapped the base in a fresh (untrained) adapter;
            # reload from the checkpoint instead so evaluation uses trained weights.
            model = model.get_base_model() if hasattr(model, "get_base_model") else model
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        info["adapter_path"] = adapter_path
        if cfg.evaluation.backend.merge_adapter:
            model = model.merge_and_unload()
            info["adapter_merged"] = True
    model.eval()
    return model, info


def resolve_device(model, requested: str | None = None):
    import torch

    if requested:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive
        return torch.device("cpu")
