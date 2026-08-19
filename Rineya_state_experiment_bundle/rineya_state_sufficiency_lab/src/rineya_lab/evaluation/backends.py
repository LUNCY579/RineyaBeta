"""Generation backends.

``hf``        -- a real transformers model, optionally with a trained adapter.
``reference`` -- a deterministic, dependency-free backend used to exercise the
                 entire evaluation stack offline (CI, smoke tests, harness
                 debugging). It can emit perfect answers, formatting noise,
                 malformed JSON, refusals, empty strings, loops and
                 non-terminating rollouts, so every branch of the parsing and
                 failure-taxonomy code is reachable without a GPU.

The reference backend is a *test fixture*. Any report generated with it is
labelled as such and must never be presented as a result about the hypothesis.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rineya_lab.config import BackendConfig, GenerationConfig, LabConfig
from rineya_lab.utils.hashing import sha256_text
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("evaluation.backends")


@dataclass
class GenerationResult:
    text: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Backend:
    """Interface implemented by every backend."""

    name = "abstract"
    is_reference = False

    def generate(self, messages: list[dict[str, str]], **overrides: Any) -> GenerationResult:
        raise NotImplementedError

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        oracles: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> list[GenerationResult]:
        """Generate for several prompts at once.

        ``oracles`` carries the per-item gold context the reference backend needs;
        real backends ignore it. It is a parallel list, not a shared value, so a
        batch cannot accidentally score every item against one item's gold data.
        """
        return [self.generate(messages, **overrides) for messages in batch]

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "is_reference": self.is_reference}

    def close(self) -> None:  # pragma: no cover - optional cleanup
        return None


# --------------------------------------------------------------------------------------
# Hugging Face backend
# --------------------------------------------------------------------------------------


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg: LabConfig) -> None:
        from rineya_lab.training.chat import ChatFormatter, ensure_pad_token
        from rineya_lab.training.modeling import build_tokenizer, load_model_for_inference, resolve_device

        self.cfg = cfg
        self.gen_cfg: GenerationConfig = cfg.evaluation.generation
        self.tokenizer = build_tokenizer(cfg.model)
        ensure_pad_token(self.tokenizer)
        # Left padding keeps generated continuations aligned in a batch.
        self.tokenizer.padding_side = "left"
        self.model, self.model_info = load_model_for_inference(cfg, cfg.evaluation.backend.adapter_path)
        self.device = resolve_device(self.model, cfg.evaluation.backend.device)
        self.formatter = ChatFormatter(self.tokenizer, cfg.model)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "is_reference": False,
            "model": self.model_info,
            "generation": vars(self.gen_cfg).copy(),
            "stop_token_ids": self._stop_token_ids(),
            "chat_template": self.formatter.template_info(),
        }

    def _stop_token_ids(self) -> list[int]:
        """Every id the checkpoint considers a stop token.

        Chat checkpoints routinely declare more than one (e.g. <|im_end|> and
        <|endoftext|>). Passing only the tokenizer's single eos lets generation run
        past a real stop, which inflates the nontermination rate and would be
        misread as a rollout-control failure.
        """
        ids: list[int] = []
        declared = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if isinstance(declared, int):
            ids.append(declared)
        elif isinstance(declared, (list, tuple)):
            ids.extend(int(value) for value in declared)
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        seen: list[int] = []
        for value in ids:
            if value not in seen:
                seen.append(value)
        return seen

    def _generation_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        cfg = self.gen_cfg
        stop_ids = self._stop_token_ids()
        kwargs: dict[str, Any] = {
            "max_new_tokens": overrides.get("max_new_tokens", cfg.max_new_tokens),
            "do_sample": overrides.get("do_sample", cfg.do_sample),
            "repetition_penalty": cfg.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": stop_ids if len(stop_ids) > 1 else (stop_ids[0] if stop_ids else None),
        }
        if kwargs["do_sample"]:
            kwargs.update({"temperature": cfg.temperature, "top_p": cfg.top_p})
            if cfg.top_k:
                kwargs["top_k"] = cfg.top_k
        if cfg.stop_strings:
            kwargs["stop_strings"] = list(cfg.stop_strings)
            kwargs["tokenizer"] = self.tokenizer
        return kwargs

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        oracles: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> list[GenerationResult]:
        import torch

        if not batch:
            return []
        prompts = [self.formatter.render_prompt(messages) for messages in batch]
        generation_kwargs = self._generation_kwargs(overrides)
        max_new_tokens = generation_kwargs["max_new_tokens"]
        try:
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            if self.gen_cfg.seed is not None:
                torch.manual_seed(self.gen_cfg.seed)
            with torch.no_grad():
                output = self.model.generate(**encoded, **generation_kwargs)
            prompt_length = encoded["input_ids"].shape[1]
            completions = output[:, prompt_length:]
            texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
            stop_ids = set(self._stop_token_ids())
            results = []
            for index, text in enumerate(texts):
                row = completions[index]
                # Count up to and including the first EOS. Counting non-pad tokens
                # would undercount whenever PAD is aliased to EOS, which is the
                # common case for decoder-only checkpoints.
                stop_mask = None
                for stop_id in stop_ids:
                    hit = row == stop_id
                    stop_mask = hit if stop_mask is None else (stop_mask | hit)
                stop_positions = stop_mask.nonzero() if stop_mask is not None else None
                if stop_positions is not None and stop_positions.numel() > 0:
                    generated = int(stop_positions[0].item()) + 1
                    finish_reason = "stop"
                else:
                    generated = int(row.shape[0])
                    finish_reason = "length" if generated >= max_new_tokens else "stop"
                results.append(
                    GenerationResult(
                        text=text,
                        prompt_tokens=prompt_length,
                        completion_tokens=generated,
                        finish_reason=finish_reason,
                    )
                )
            return results
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            LOGGER.error("Generation failed: %s", exc)
            return [GenerationResult(text=None, error=f"{type(exc).__name__}: {exc}", finish_reason="error") for _ in batch]

    def generate(self, messages: list[dict[str, str]], **overrides: Any) -> GenerationResult:
        return self.generate_batch([messages], **overrides)[0]

    def close(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover - GPU only
                torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Reference backend
# --------------------------------------------------------------------------------------

REFERENCE_MODES = (
    "oracle",            # always the canonical next step / answer, contract-compliant
    "oracle_prose",      # correct content, prose formatting (exercises lenient track)
    "noisy",             # seeded mixture of correct, malformed, refusal and empty
    "malformed_json",    # valid content wrapped in broken JSON (repair path)
    "refusal",           # always refuses
    "empty",             # always empty
    "looping",           # repeats the same state forever (loop detector)
    "stagnant",          # never changes the state (stagnation detector)
    "premature",         # claims done immediately with no answer
    "nonterminating",    # never sets done
    "wrong",             # confidently wrong answers
)


class ReferenceBackend(Backend):
    """Deterministic scripted backend driven by the gold data in the prompt context."""

    name = "reference"
    is_reference = True

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        if cfg.reference_mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown reference_mode {cfg.reference_mode!r}; expected one of {REFERENCE_MODES}")
        self.mode = cfg.reference_mode
        self.noise = cfg.reference_noise
        self.seed = cfg.reference_seed
        self._oracle: dict[str, Any] = {}
        self._call_count = 0
        # Per-problem call counters. Behaviour must depend on a problem's OWN
        # progress, not on global call order, or batching would change what the
        # fixture does and the harness tests would stop being reproducible.
        self._problem_calls: dict[str, int] = {}

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "is_reference": True,
            "reference_mode": self.mode,
            "reference_noise": self.noise,
            "reference_seed": self.seed,
            "warning": "Reference backend output is synthetic; it validates the harness, not the hypothesis.",
        }

    def set_oracle(self, oracle: dict[str, Any]) -> None:
        """Supply the gold context for the next call (protocol runners do this)."""
        self._oracle = oracle or {}

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        oracles: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> list[GenerationResult]:
        if oracles is not None and len(oracles) != len(batch):
            raise ValueError("oracles must be a parallel list to batch")
        results: list[GenerationResult] = []
        for index, messages in enumerate(batch):
            if oracles is not None:
                self.set_oracle(oracles[index])
            results.append(self.generate(messages, **overrides))
        return results

    def _roll(self, key: str) -> float:
        """Deterministic per-item draw.

        Keyed only on the item identity, so the same item behaves identically at
        any batch size. Mixing in a global counter would make the noisy profile
        depend on how work happened to be grouped.
        """
        digest = sha256_text(f"{self.seed}|{key}")
        return int(digest[:8], 16) / 0xFFFFFFFF

    def generate(self, messages: list[dict[str, str]], **overrides: Any) -> GenerationResult:
        self._call_count += 1
        prompt = messages[-1]["content"]
        stateful = "Return a JSON object" in prompt
        oracle = dict(self._oracle)
        key = oracle.get("transition_id") or oracle.get("problem_id") or sha256_text(prompt)[:16]
        problem_key = oracle.get("problem_id") or key
        visit = self._problem_calls.get(problem_key, 0)
        self._problem_calls[problem_key] = visit + 1

        mode = self.mode
        if mode == "noisy":
            roll = self._roll(key)
            if roll < self.noise * 0.4:
                mode = "malformed_json"
            elif roll < self.noise * 0.6:
                mode = "refusal"
            elif roll < self.noise * 0.8:
                mode = "empty"
            elif roll < self.noise:
                mode = "wrong"
            else:
                mode = "oracle"

        text = self._render(mode, stateful, oracle, prompt, visit)
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()) if text else 0,
            finish_reason="stop",
            meta={"reference_mode": mode},
        )

    def _render(self, mode: str, stateful: bool, oracle: dict[str, Any], prompt: str, visit: int = 0) -> str | None:
        step = oracle.get("target_step", "Perform the next operation.")
        next_state = oracle.get("next_state", "State updated.")
        done = bool(oracle.get("done", False))
        answer = oracle.get("final_answer") or oracle.get("answer")
        full_response = oracle.get("full_response")

        if mode == "empty":
            return ""
        if mode == "refusal":
            return "I'm sorry, I cannot help with this request."
        if mode == "wrong":
            wrong_answer = _perturb_answer(str(answer)) if answer else "unknown"
            if stateful:
                return json.dumps(
                    {
                        "next_reasoning_step": "Apply an incorrect operation.",
                        "updated_compact_state": f"Incorrect state: {wrong_answer}.",
                        "done": done,
                        "final_answer": wrong_answer if done else None,
                    }
                )
            return f"Final answer: {wrong_answer}"
        if mode == "premature":
            return json.dumps({"next_reasoning_step": step, "updated_compact_state": next_state, "done": True,
                               "final_answer": None}) if stateful else "Final answer:"
        if mode == "nonterminating":
            # A state that always advances and never repeats: no loop, no stagnation,
            # no completion -- the rollout can only end by exhausting its step budget.
            state = f"{next_state} (iteration {visit})"
            return json.dumps({"next_reasoning_step": step, "updated_compact_state": state, "done": False,
                               "final_answer": None}) if stateful else step
        if mode == "stagnant":
            state = oracle.get("current_state", "State unchanged.")
            return json.dumps({"next_reasoning_step": step, "updated_compact_state": state, "done": False,
                               "final_answer": None}) if stateful else step
        if mode == "looping":
            # Cycle between two fixed states: the state keeps changing (so the
            # stagnation detector stays quiet) but revisits a previous value, which is
            # exactly what the loop detector must catch.
            tag = oracle.get("problem_id", "problem")
            state = f"Loop state {(visit % 2) + 1} for {tag}."
            return json.dumps({"next_reasoning_step": step, "updated_compact_state": state, "done": False,
                               "final_answer": None}) if stateful else step
        if mode == "malformed_json":
            payload = (
                '{"next_reasoning_step": "' + str(step).replace('"', "'") + '", '
                '"updated_compact_state": "' + str(next_state).replace('"', "'") + '", '
                '"done": ' + ("True" if done else "False") + ', '
                '"final_answer": ' + (f'"{answer}"' if done and answer else "None") + ','
            )
            return "Here is the transition:\n" + payload
        if mode == "oracle_prose":
            if stateful:
                return (
                    f"Next I will: {step}\n"
                    f"Updated state: {next_state}\n"
                    f"Done: {'yes' if done else 'no'}\n"
                    + (f"The answer is {answer}." if done and answer else "")
                )
            return f"{step}" if not done else f"{step}\nThe answer is {answer}."

        # oracle
        if stateful:
            return json.dumps(
                {
                    "next_reasoning_step": step,
                    "updated_compact_state": next_state,
                    "done": done,
                    "final_answer": str(answer) if done and answer is not None else None,
                },
                ensure_ascii=False,
            )
        if full_response is not None and "Current compact state" not in prompt:
            return str(full_response)
        if answer is not None and "Give only the final answer" in prompt:
            return f"Final answer: {answer}"
        return str(step)


def _perturb_answer(answer: str) -> str:
    """Produce a confidently wrong but type-plausible answer."""
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if match:
        value = match.group(0)
        try:
            perturbed = str(int(value) + 1) if "." not in value else f"{float(value) + 1:.2f}"
        except ValueError:  # pragma: no cover - defensive
            perturbed = value
        return answer[: match.start()] + perturbed + answer[match.end():]
    lowered = answer.strip().lower()
    if lowered in ("true", "false"):
        return "False" if lowered == "true" else "True"
    return answer + " (incorrect)"


def build_backend(cfg: LabConfig) -> Backend:
    kind = cfg.evaluation.backend.kind
    if kind == "reference":
        return ReferenceBackend(cfg.evaluation.backend)
    if kind == "hf":
        return HFBackend(cfg)
    raise ValueError(f"Unknown backend kind {kind!r}")
