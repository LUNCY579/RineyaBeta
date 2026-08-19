"""The SFT training loop shared by every condition.

Written as an explicit loop rather than ``Trainer`` because the experiment needs
controls a generic trainer does not expose:

* a fully reproducible, externally planned example order (the order ablation);
* per-example loss weights (``inverse_transitions``);
* token-exact budget stopping (Regime T) and update-exact stopping (Regime U);
* exposure accounting that separates input, target and padded tokens;
* resume that restores the exact position inside a planned epoch.

FULL and SPLIT run through this identical code path. The only difference between
the two arms is the ``data_view`` the plan was built from.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rineya_lab.config import LabConfig
from rineya_lab.data.budget import BudgetGuard, ExposureAccount
from rineya_lab.data.loading import RineyaDataset
from rineya_lab.data.views import ViewPlan, build_view_plan, iter_examples, order_diagnostics
from rineya_lab.training.chat import ChatFormatter, ensure_pad_token
from rineya_lab.training.collate import batched, pad_batch, to_torch_batch, weighted_causal_lm_loss
from rineya_lab.training.schedule import build_lambda, resolve_warmup_steps
from rineya_lab.utils.determinism import seed_everything, warn_if_hash_seed_unpinned
from rineya_lab.utils.envinfo import collect_environment
from rineya_lab.utils.io import write_json
from rineya_lab.utils.logging import get_logger, setup_logging
from rineya_lab.version import __version__

LOGGER = get_logger("training.loop")

STATE_FILENAME = "training_state.json"
OPTIMIZER_FILENAME = "optimizer.pt"


@dataclass
class DevLossProbe:
    """A fixed held-out batch used only to choose a checkpoint.

    This is DEV, not TEST, and it is used for exactly one decision -- which
    adapter to evaluate -- under a rule that is identical for both arms and
    recorded in the manifest.
    """

    encodings: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    problem_ids: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.encodings)


@dataclass
class TrainingState:
    epoch: int = 0
    position: int = 0
    global_micro_step: int = 0
    optimizer_updates: int = 0
    finished: bool = False
    stop_reason: str | None = None
    account: dict[str, Any] = field(default_factory=dict)
    loss_history: list[dict[str, Any]] = field(default_factory=list)
    dev_loss_history: list[dict[str, Any]] = field(default_factory=list)
    best_dev_loss: float | None = None
    best_dev_update: int | None = None


def _load_tokenizer(cfg: LabConfig):
    from rineya_lab.training.stub_tokenizer import ByteTokenizer, is_builtin

    if is_builtin(cfg.model.tokenizer_name_or_path or cfg.model.name_or_path):
        LOGGER.warning("Using the builtin byte tokenizer; this is a smoke-test fixture, not a research setting")
        return ByteTokenizer()
    from rineya_lab.training.modeling import build_tokenizer

    return build_tokenizer(cfg.model)


def _encode_stream(dataset, plan, cfg, formatter, account, guard, *, start_epoch, start_position):
    """Yield (example, encoded) pairs, applying unique-problem admission control."""
    for example in iter_examples(dataset, plan, cfg, start_epoch=start_epoch, start_position=start_position):
        if not guard.accepts_more_problems(account, example.problem_id):
            account.dropped_records += 1
            continue
        encoded = formatter.encode_example(
            example.messages,
            max_seq_len=cfg.training.max_seq_len,
            prompt_head_keep=cfg.training.prompt_head_keep,
        )
        yield example, encoded


def _build_dev_probe(dataset, cfg: LabConfig, formatter) -> DevLossProbe:
    """Encode a deterministic slice of the DEV split in the run's own view."""
    from rineya_lab.data.views import build_view_plan, iter_examples

    probe_cfg = copy.deepcopy(cfg)
    probe_cfg.training.epochs = 1.0
    probe_cfg.data_view.max_records = cfg.training.dev_eval_max_records
    probe_cfg.data_view.one_transition_per_problem = False
    probe = DevLossProbe()
    plan = build_view_plan(dataset, probe_cfg, split=cfg.dataset.dev_split)
    for example in iter_examples(dataset, plan, probe_cfg):
        probe.encodings.append(
            formatter.encode_example(
                example.messages,
                max_seq_len=cfg.training.max_seq_len,
                prompt_head_keep=cfg.training.prompt_head_keep,
            )
        )
        probe.weights.append(example.weight)
        probe.problem_ids.append(example.problem_id)
    LOGGER.info("DEV loss probe: %d records from split %s", len(probe), cfg.dataset.dev_split)
    return probe


def _dev_loss(model, probe: DevLossProbe, tokenizer, cfg: LabConfig, device) -> float:
    """Mean per-example target NLL over the probe. Lower is better."""
    import torch

    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    batch_size = max(1, cfg.training.dev_eval_batch_size)
    with torch.no_grad():
        for start in range(0, len(probe), batch_size):
            encodings = probe.encodings[start : start + batch_size]
            batch = pad_batch(
                encodings,
                pad_token_id=tokenizer.pad_token_id,
                weights=[1.0] * len(encodings),
                problem_ids=probe.problem_ids[start : start + batch_size],
            )
            tensors = to_torch_batch(batch, device=device)
            outputs = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"])
            _, meta = weighted_causal_lm_loss(outputs.logits, tensors["labels"], tensors["weights"])
            per_example = meta["per_example_loss"]
            total += float(per_example.sum().item())
            count += int(per_example.numel())
    if was_training:
        model.train()
    return total / max(1, count)


def _planned_updates(plan: ViewPlan, cfg: LabConfig) -> int:
    effective = cfg.training.per_device_batch_size * cfg.training.gradient_accumulation_steps
    natural = max(1, plan.total_rows // effective)
    if cfg.budget.max_optimizer_updates is not None:
        return min(natural, cfg.budget.max_optimizer_updates)
    return natural


def dry_run(cfg: LabConfig, dataset: RineyaDataset) -> dict[str, Any]:
    """Walk the entire planned stream, encoding and accounting, without a model.

    This validates the data path, the chat template, the truncation policy and
    the budget arithmetic on a CPU-only machine before any GPU time is spent.
    """
    tokenizer = _load_tokenizer(cfg)
    ensure_pad_token(tokenizer)
    formatter = ChatFormatter(tokenizer, cfg.model)
    plan = build_view_plan(dataset, cfg)
    account = ExposureAccount()
    guard = BudgetGuard(cfg.budget)

    effective_batch = cfg.training.per_device_batch_size * cfg.training.gradient_accumulation_steps
    micro = 0
    for example, encoded in _encode_stream(
        dataset, plan, cfg, formatter, account, guard, start_epoch=0, start_position=0
    ):
        account.observe_example(
            problem_id=example.problem_id,
            input_tokens=encoded.prompt_token_count,
            target_tokens=encoded.target_token_count,
            weight=example.weight,
            prompt_truncated=encoded.prompt_truncated,
            target_truncated=encoded.target_truncated,
        )
        micro += 1
        if micro % cfg.training.per_device_batch_size == 0:
            account.observe_batch(padded_tokens=0)
        if account.records_seen % effective_batch == 0:
            account.observe_update()
        if guard.should_stop(account):
            break

    return {
        "mode": "dry_run",
        "plan": plan.summary(),
        "order_diagnostics": order_diagnostics(plan),
        "exposure": account.as_dict(),
        "template": formatter.template_info(),
        "stop_reason": guard.stop_reason,
    }


def train(cfg: LabConfig, dataset: RineyaDataset | None = None) -> dict[str, Any]:
    """Train one adapter under one condition. Returns the run manifest."""
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=output_dir / "train.log")

    seed_record = seed_everything(cfg.experiment.seed)
    hash_warning = warn_if_hash_seed_unpinned()
    if hash_warning:
        LOGGER.warning(hash_warning)

    dataset = dataset or RineyaDataset.from_config(cfg.dataset)

    manifest: dict[str, Any] = {
        "lab_version": __version__,
        "experiment": asdict(cfg.experiment),
        "resolved_config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "seeding": seed_record,
        "environment": collect_environment(),
        "dataset": dataset.provenance(splits=(cfg.dataset.train_split,)),
        "started_at": time.time(),
    }

    if cfg.training.dry_run:
        report = dry_run(cfg, dataset)
        manifest["result"] = report
        manifest["finished_at"] = time.time()
        write_json(output_dir / "run_manifest.json", manifest)
        LOGGER.info("Dry run complete: %s", report["exposure"])
        return manifest

    import torch

    from rineya_lab.training.modeling import build_model, resolve_device

    tokenizer = _load_tokenizer(cfg)
    ensure_pad_token(tokenizer)
    formatter = ChatFormatter(tokenizer, cfg.model)
    manifest["chat_template"] = formatter.template_info()

    plan = build_view_plan(dataset, cfg)
    manifest["view_plan"] = plan.summary()
    manifest["order_diagnostics"] = order_diagnostics(plan)

    model, model_info = build_model(cfg, for_training=True)
    manifest["model"] = model_info
    device = resolve_device(model, cfg.training.device or cfg.evaluation.backend.device)
    if not cfg.model.device_map:
        model.to(device)
    model.train()

    total_updates = _planned_updates(plan, cfg)
    warmup = resolve_warmup_steps(total_updates, cfg.training.warmup_ratio, cfg.training.warmup_steps)
    decay_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        decay_params,
        lr=cfg.training.learning_rate,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_epsilon,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lambda(total_updates, warmup, cfg.training.lr_scheduler)
    )
    manifest["optimization"] = {
        "planned_optimizer_updates": total_updates,
        "warmup_updates": warmup,
        "scheduler": cfg.training.lr_scheduler,
        "peak_learning_rate": cfg.training.learning_rate,
        "effective_batch_size": cfg.training.per_device_batch_size * cfg.training.gradient_accumulation_steps,
        "optimizer": "AdamW",
    }

    dev_probe: DevLossProbe | None = None
    if cfg.training.eval_steps > 0:
        dev_probe = _build_dev_probe(dataset, cfg, formatter)
        manifest["dev_probe"] = {
            "split": cfg.dataset.dev_split,
            "records": len(dev_probe),
            "unique_problems": len(set(dev_probe.problem_ids)),
            "purpose": "checkpoint selection only",
        }
    manifest["checkpoint_selection"] = {
        "rule": cfg.training.checkpoint_selection,
        "eval_steps": cfg.training.eval_steps,
    }

    state = TrainingState()
    if cfg.training.resume_from:
        state = _restore(cfg.training.resume_from, model, optimizer, scheduler)
        LOGGER.info("Resumed at epoch %d position %d (update %d)", state.epoch, state.position, state.optimizer_updates)

    account = ExposureAccount()
    account.optimizer_updates = state.optimizer_updates
    guard = BudgetGuard(cfg.budget)
    scaler = torch.amp.GradScaler("cuda", enabled=model_info["precision_resolved"] == "fp16" and device.type == "cuda")

    started = time.time()
    micro_batch_size = cfg.training.per_device_batch_size
    accum = cfg.training.gradient_accumulation_steps
    stream = _encode_stream(
        dataset, plan, cfg, formatter, account, guard,
        start_epoch=state.epoch, start_position=state.position,
    )

    pending = 0
    stop = False
    last_example = None
    for chunk in batched(stream, micro_batch_size):
        examples = [item[0] for item in chunk]
        encodings = [item[1] for item in chunk]
        last_example = examples[-1]
        for example, encoded in chunk:
            account.observe_example(
                problem_id=example.problem_id,
                input_tokens=encoded.prompt_token_count,
                target_tokens=encoded.target_token_count,
                weight=example.weight,
                prompt_truncated=encoded.prompt_truncated,
                target_truncated=encoded.target_truncated,
            )
        batch = pad_batch(
            encodings,
            pad_token_id=tokenizer.pad_token_id,
            weights=[e.weight for e in examples],
            example_ids=[e.example_id for e in examples],
            problem_ids=[e.problem_id for e in examples],
        )
        account.observe_batch(padded_tokens=batch.padded_tokens)
        tensors = to_torch_batch(batch, device=device)

        autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(model_info["precision_resolved"])
        context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None and device.type in ("cuda", "cpu")
            else _nullcontext()
        )
        with context:
            outputs = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"])
            loss, loss_meta = weighted_causal_lm_loss(
                outputs.logits, tensors["labels"], tensors["weights"], label_smoothing=cfg.training.label_smoothing
            )
        scaled = loss / accum
        if scaler.is_enabled():
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        pending += 1
        state.global_micro_step += 1
        if pending == accum:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decay_params, cfg.training.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            account.observe_update()
            state.optimizer_updates = account.optimizer_updates

            if account.optimizer_updates % max(1, cfg.training.logging_steps) == 0:
                entry = {
                    "update": account.optimizer_updates,
                    "loss": float(loss.detach().item()),
                    "lr": scheduler.get_last_lr()[0],
                    "records_seen": account.records_seen,
                    "target_tokens": account.target_tokens,
                    "supervised_tokens_in_batch": loss_meta["supervised_tokens"],
                    "elapsed_s": round(time.time() - started, 2),
                }
                state.loss_history.append(entry)
                LOGGER.info(
                    "update=%d loss=%.4f lr=%.3e records=%d target_tokens=%d",
                    entry["update"], entry["loss"], entry["lr"], entry["records_seen"], entry["target_tokens"],
                )
            if dev_probe is not None and account.optimizer_updates % cfg.training.eval_steps == 0:
                # Rounded once, then used for both the history and the tracked best,
                # so a reader cannot find best_dev_loss absent from dev_loss_history.
                dev_value = round(_dev_loss(model, dev_probe, tokenizer, cfg, device), 6)
                state.dev_loss_history.append({"update": account.optimizer_updates, "dev_loss": dev_value})
                improved = state.best_dev_loss is None or dev_value < state.best_dev_loss
                LOGGER.info(
                    "update=%d dev_loss=%.4f%s",
                    account.optimizer_updates, dev_value, " (best)" if improved else "",
                )
                if improved:
                    state.best_dev_loss = dev_value
                    state.best_dev_update = account.optimizer_updates
                    if cfg.training.checkpoint_selection == "best_dev_loss":
                        _save_adapter(output_dir / "best_adapter", model, tokenizer)

            if cfg.training.save_steps and account.optimizer_updates % cfg.training.save_steps == 0:
                state.epoch = last_example.epoch
                state.position = last_example.position + 1
                state.account = account.as_dict()
                _save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, state, cfg)

            if account.optimizer_updates >= total_updates:
                guard.stop_reason = guard.stop_reason or f"planned optimizer updates reached ({total_updates})"
                stop = True

        if guard.should_stop(account) or stop:
            stop = True
            break

    if pending > 0 and not stop:
        # flush a partial accumulation group so no supervised tokens are discarded
        torch.nn.utils.clip_grad_norm_(decay_params, cfg.training.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        account.observe_update()

    state.finished = True
    state.stop_reason = guard.stop_reason or "data exhausted"
    state.account = account.as_dict()
    if last_example is not None:
        state.epoch = last_example.epoch
        state.position = last_example.position + 1

    final_dir = output_dir / "adapter"
    _save_adapter(final_dir, model, tokenizer)
    selected = "last"
    if cfg.training.checkpoint_selection == "best_dev_loss" and (output_dir / "best_adapter").exists():
        # The evaluated adapter is the best-DEV one; the final-step adapter is kept
        # beside it as last_adapter so the choice is auditable rather than lost.
        shutil.rmtree(output_dir / "last_adapter", ignore_errors=True)
        shutil.copytree(final_dir, output_dir / "last_adapter")
        shutil.rmtree(final_dir)
        shutil.copytree(output_dir / "best_adapter", final_dir)
        selected = f"best_dev_loss@update_{state.best_dev_update}"
    _write_state(output_dir, state)

    wall_clock = time.time() - started
    manifest["training"] = {
        "stop_reason": state.stop_reason,
        "wall_clock_seconds": round(wall_clock, 3),
        "exposure": account.as_dict(),
        "loss_history": state.loss_history,
        "final_epoch": state.epoch,
        "final_position": state.position,
        "records_per_second": round(account.records_seen / wall_clock, 3) if wall_clock > 0 else None,
        "target_tokens_per_second": round(account.target_tokens / wall_clock, 3) if wall_clock > 0 else None,
        "peak_memory_bytes": _peak_memory(),
        "adapter_dir": str(final_dir),
        "dev_loss_history": state.dev_loss_history,
        "best_dev_loss": state.best_dev_loss,
        "best_dev_update": state.best_dev_update,
        "selected_checkpoint": selected,
    }
    manifest["finished_at"] = time.time()
    write_json(output_dir / "run_manifest.json", manifest)
    LOGGER.info("Training complete: %s", manifest["training"]["exposure"])
    return manifest


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _peak_memory() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - GPU only
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def _save_adapter(directory: Path, model, tokenizer) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(directory))
    if hasattr(tokenizer, "save_pretrained"):
        try:
            tokenizer.save_pretrained(str(directory))
        except Exception as exc:  # pragma: no cover - stub tokenizers
            LOGGER.warning("Could not save tokenizer alongside the adapter: %s", exc)


def _write_state(output_dir: Path, state: TrainingState) -> None:
    write_json(output_dir / STATE_FILENAME, asdict(state))


def _save_checkpoint(output_dir: Path, model, tokenizer, optimizer, scheduler, state: TrainingState, cfg: LabConfig) -> None:
    import torch

    checkpoint_dir = output_dir / "checkpoints" / f"update_{state.optimizer_updates:07d}"
    _save_adapter(checkpoint_dir, model, tokenizer)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / OPTIMIZER_FILENAME,
    )
    _write_state(checkpoint_dir, state)
    LOGGER.info("Saved checkpoint %s", checkpoint_dir)
    _prune_checkpoints(output_dir / "checkpoints", cfg.training.save_total_limit)


def _prune_checkpoints(root: Path, keep: int) -> None:
    if keep <= 0 or not root.exists():
        return
    checkpoints = sorted(p for p in root.iterdir() if p.is_dir())
    for stale in checkpoints[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)
        LOGGER.info("Pruned old checkpoint %s", stale)


def _restore(resume_from: str, model, optimizer, scheduler) -> TrainingState:
    import torch
    from peft import PeftModel

    checkpoint_dir = Path(resume_from)
    state_path = checkpoint_dir / STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"No {STATE_FILENAME} in {checkpoint_dir}; cannot resume deterministically")
    state = TrainingState(**json.loads(state_path.read_text(encoding="utf-8")))

    optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
    if optimizer_path.exists():
        payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])

    if isinstance(model, PeftModel):
        model.load_adapter(str(checkpoint_dir), adapter_name="default", is_trainable=True)
    state.finished = False
    return state


def format_manifest_summary(manifest: dict[str, Any]) -> str:
    """One-screen human summary printed at the end of a run."""
    training = manifest.get("training") or manifest.get("result", {})
    exposure = training.get("exposure", {})
    lines = [
        f"condition        : {manifest['experiment']['condition']}",
        f"name             : {manifest['experiment']['name']}",
        f"seed             : {manifest['experiment']['seed']}",
        f"stop reason      : {training.get('stop_reason')}",
        f"records seen     : {exposure.get('records_seen')}",
        f"unique problems  : {exposure.get('unique_problems_seen')}",
        f"input tokens     : {exposure.get('input_tokens')}",
        f"target tokens    : {exposure.get('target_tokens')}",
        f"padded tokens    : {exposure.get('padded_tokens')}",
        f"optimizer updates: {exposure.get('optimizer_updates')}",
        f"truncated prompts: {exposure.get('truncated_prompts')}",
        f"truncated targets: {exposure.get('truncated_targets')}",
    ]
    if "wall_clock_seconds" in training:
        lines.append(f"wall clock (s)   : {training['wall_clock_seconds']}")
    return "\n".join(lines)
