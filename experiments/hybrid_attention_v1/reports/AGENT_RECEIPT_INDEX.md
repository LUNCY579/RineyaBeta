# Index of preserved receipts (agent continuation)

Current source `code_identity` is `57bb7678183d84ac...`. A receipt only gates anything when its
`code_identity` equals that. Everything else is preserved as a record of how the experiment
reached its current state, NOT as a live authorisation.

| receipt | stage | verdict | code_identity | criterion/readout | status |
|---|---|---|---|---|---|
| `agent_p0_offline_receipt.json` | cpu_fixture_qualification | PASS | `6ac74b664eb5` | - | superseded code |
| `agent_p1_checkpoint_audit_receipt.json` | local_checkpoint_audit | PASS | `6ac74b664eb5` | - | superseded code |
| `agent_p1_checkpoint_inventory.json` | checkpoint inventory | - | `-` | - | asset identity for P1 |
| `agent_p2_cpu_math_receipt.json` | real_model_qualification | FAIL | `1f8373e6ba1d` | - | superseded code |
| `agent_p2_cpu_sdpa_receipt.json` | real_model_qualification | FAIL | `1f8373e6ba1d` | - | superseded code |
| `agent_p2_cuda_math_receipt.json` | real_model_qualification | FAIL | `1f8373e6ba1d` | - | superseded code |
| `agent_p2_cuda_sdpa_receipt.json` | real_model_qualification | FAIL | `1f8373e6ba1d` | - | superseded code |
| `agent_p2_v2_cpu_math_receipt.json` | real_model_qualification | PASS | `998bf6951b2b` | crit v2 | superseded code |
| `agent_p2_v2_cpu_sdpa_receipt.json` | real_model_qualification | PASS | `998bf6951b2b` | crit v2 | superseded code |
| `agent_p2_v2_cuda_math_receipt.json` | real_model_qualification | PASS | `998bf6951b2b` | crit v2 | superseded code |
| `agent_p2_v2_cuda_sdpa_receipt.json` | real_model_qualification | PASS | `998bf6951b2b` | crit v2 | superseded code |
| `agent_p3_final_gate_receipt.json` | real_model_qualification | PASS | `57bb7678183d` | crit v2 | **CURRENT CODE** |
| `agent_p3_final_receipt.json` | open_synthetic_diagnostic | PASS | `57bb7678183d` | readout v2 | **CURRENT CODE** |
| `agent_p3_final_summary.json` | ? | FAIL | `-` | - | superseded code |
| `agent_p3_receipt.json` | open_synthetic_diagnostic | PASS | `0718c7a1c019` | - | superseded code |
| `agent_p3_summary.json` | ? | FAIL | `-` | - | superseded code |

## How to read this

* The LIVE chain is P0 -> P1 -> P2 (`agent_p3_final_gate_receipt.json`) -> P3
  (`agent_p3_final_receipt.json`), all at the current identity. Only these authorise anything,
  and the P3 results in `AGENT_P3_RESULTS.md` are read off that P3 receipt.
* `agent_p2_*_receipt.json` (four, identity `1f8373e6`) are the ORIGINAL P2 attempt and they
  FAILED, under the superseded v1 absolute-tolerance criterion. Kept because the failure is
  the finding -- see `AGENT_REAL_MODEL_QUALIFICATION.md`. NOT a qualification of the retrofit.
* `agent_p2_v2_*_receipt.json` (four, identity `998bf695`) are those same four device/backend
  combinations re-judged under criterion v2, all PASS, on earlier code.
* `agent_p3_receipt.json` (identity `0718c7a1`) is the FIRST P3 run. It passed as a run but its
  capability endpoint floored at 0.000 on every arm -- see `AGENT_P3_DIAGNOSTIC.md`. It is
  superseded and must NOT be quoted as a capability result.

Stage reports: `AGENT_REAL_MODEL_QUALIFICATION.md` (P1/P2), `AGENT_P3_DIAGNOSTIC.md` (the
floored first P3 and the readout amendment), `AGENT_P3_RESULTS.md` (the final P3 result and
the corrections an adversarial audit forced), `AGENT_PROPOSED_P2_AMENDMENT.md` (the criterion
amendment, since adopted as protocol `qualification.criterion_version` 2).
