# Pinned sources and claim boundaries

## Repository basis, verified before branch creation

- RineyaBeta main: `3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`.
  https://github.com/LUNCY579/RineyaBeta/blob/3532c7bc8dbdd48ef8801ba6080cb80b0e90379a/docs/RESEARCH_PROGRAM.md
  Supplies the governing principles and distinguishes the 474,021,888-parameter
  vocabulary-trimmed R2 from Qwen3-0.6B. Parameter counts remain source-reported
  until the actual local checkpoint is loaded and counted here.
- R2 config at that same commit:
  https://github.com/LUNCY579/RineyaBeta/blob/3532c7bc8dbdd48ef8801ba6080cb80b0e90379a/artifacts/rineya-core-en32k-28L/config.json
  28 layers, hidden 1024, MLP 3072, 16Q/8KV, head dimension 128, vocabulary 32768,
  BF16 config, raw EOS 32744. Shape is not proof of local weight lineage.
- Historical routing and protected training fingerprint:
  https://github.com/LUNCY579/RineyaBeta/blob/3532c7bc8dbdd48ef8801ba6080cb80b0e90379a/docs/README.md
  No scripts/train, prior experiments, datasets, checkpoints or historical reports
  are changed by this programme.
- Separate HLGA branch observed at `a196318ec75cacccc8ac9ec446e4c4b6a29c2802`:
  https://github.com/LUNCY579/RineyaBeta/blob/a196318ec75cacccc8ac9ec446e4c4b6a29c2802/experiments/hybrid_attention_v1/protocol.json
  Its compatibility and recovery contract is not inherited by QRR.
- Cornerstone branch observed at `1cc2d8e2f53d937ae74641b3851b1f80b05518cd`.

## Prior adjudication inputs, not rerun in this implementation

The supplied Rineya_Research_Bundle_2026-09-08.zip, specifically AGENT_HANDOFF.md,
EXPERIMENT_CARDS.md and MEASUREMENT_CONTRACT.md, proposes E00-E09. Its suggested
promotion thresholds are not automatically approved or powered by this commit.
Its handoff originally suggested SupaExperiment; the current user specifically
requested a NEW RineyaBeta branch. This implementation uses a clean RineyaBeta
main base and a separate R2 runtime rather than merging or weakening Supa locks.

Historical Supa provenance remains at the source, not as a new QRR result:
https://github.com/LUNCY579/SupaExperiment/blob/25dac0b0584f48882823e37da205cd40c2ab7d53/experiments/nonrecurrent_adaptation_v1/EVIDENCE.md
https://github.com/LUNCY579/SupaExperiment/blob/25dac0b0584f48882823e37da205cd40c2ab7d53/experiments/nonrecurrent_adaptation_v1/ERRATA.md
https://github.com/LUNCY579/SupaExperiment/blob/4b868ff8313978657fa50c82d3584fd553fbed87/experiments/query_placement_a09_v1/RESULTS.md

The historical ~37.5% to ~86.7% raw-exact lead is not a QRR score or a composition
guarantee. A04's interaction is not adapter-versus-frozen training efficacy.
The A09 cue-copy pattern motivates the counterfactual operator instrument; it does
not establish a universal model-size limit. No historical raw prediction bank was
recovered or reconstructed in this implementation session.

## API and prior-art context

PyTorch Linear defines y=xA^T+b and its bias-free form; the prototype's residual
head seam explicitly requires the latter:
https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html

The supplied research report discusses iterative-computation precedents including
Universal Transformers (https://arxiv.org/abs/1807.03819). QRR-1 is a bounded
engineering hypothesis, not a claim to have invented iterative attention or to
have reproduced a published model's performance.
