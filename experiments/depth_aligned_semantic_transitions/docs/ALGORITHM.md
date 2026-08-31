# DAST algorithm specification

## 1. Research object

Let a decoder-only Transformer have residual hidden states
\(H^{(0)},H^{(1)},\ldots,H^{(L)}\). DAST chooses monotone checkpoint layers
\(d_1<\cdots<d_K=L\) and asks whether those checkpoints can represent a cumulative
reasoning trajectory \(S_1,\ldots,S_K\) in one shared coordinate system.

The default schedule uses normalized depth fractions. For \(L=28\) and
\((0.25,0.50,0.75,1.0)\), the exact post-layer states are 7, 14, 21, and 28.
Hidden-state index 0 is the embedding output, so the implementation reads
`hidden_states[d_k]`.

## 2. Prompt-only state readout

Training uses the same teacher-forced full-CoT target in every arm. To prevent a
trivial leak from target tokens, DAST forms a separate binary mask \(M\) containing
only system/user/generation-prompt positions. Padding and assistant positions are
zero. For mean pooling:

\[
\bar h_k=\frac{\sum_t M_t H^{(d_k)}_t}{\sum_t M_t}.
\]

The shared interface is:

\[
z_k=\operatorname{norm}\left(W_2\,\operatorname{GELU}(W_1\,\operatorname{LN}(\bar h_k))\right).
\]

The same \(W_1,W_2\) are used at every checkpoint. Independent per-layer probes
would permit incompatible coordinate systems and would not establish a semantic
boundary between blocks.

## 3. Target construction

The existing sequence-independence dataset supplies a canonical complete reasoning
trace and a compact state after every step. DAST excludes the initial
"No derived facts" state and treats post-step compact states as cumulative
milestones.

V1 target geometry is deliberately conservative:

1. tokenize each compact-state text without a chat template;
2. mean-pool the frozen base input embeddings;
3. apply a deterministic fixed projection seeded by configuration;
4. L2-normalize the result.

\[
s_k^*=\operatorname{norm}\left(\operatorname{mean}(E_{\text{frozen}}(text_k))R\right).
\]

This gives all depths a stable common target. It does **not** prove the target is a
complete symbolic representation. Multi-view paraphrase/structured targets are a
future phase contingent on V1.

## 4. Variable trajectory length

A task can have \(N\) semantic transitions while the model has \(K\) checkpoints.
DAST pins both endpoints and maps normalized progress:

\[
i_k=\operatorname{round}\left(\frac{k(N-1)}{K-1}\right),\quad k=0,\ldots,K-1.
\]

If \(N<K\), repeated states are explicit and recorded. They are not silently
removed because every arm must retain the same number of depth losses.

## 5. Arm definitions

All primary arms use identical records, full-CoT LM targets, tokenizer settings,
LoRA capacity, update budgets, and checkpoint count.

- `cot_baseline`: LM loss only; contextual baseline.
- `state_ordered`: \(d_k\leftrightarrow s_k^*\).
- `state_shuffled`: deterministic non-identity permutation per problem; same target multiset.
- `state_reverse`: \(d_k\leftrightarrow s_{K-k+1}^*\).
- `state_final_repeated`: every checkpoint predicts \(s_K^*\); generic deep-supervision control.
- `literal_ordered`: targets literal reasoning-step embeddings rather than compact semantic states.
- `operator_ordered`: ordered states plus learned transition dictionary.
- `operator_shuffled`: matched operator-capacity negative control.

The shuffled permutation is derived from SHA-256 of seed and problem ID. If the
sampled permutation is identity, it is rotated. The control therefore never
silently becomes the treatment.

## 6. State objective

The primary state loss is cosine distance:

\[
\mathcal L_{state}=\frac1K\sum_k\left(1-z_k^\top s_k^*\right).
\]

A stage-wise symmetric multi-positive InfoNCE term discourages collapse. Equal
semantic target strings in a batch are positives rather than false negatives.

\[
\mathcal L=\mathcal L_{LM}+\lambda_s\mathcal L_{state}
+\lambda_c\mathcal L_{contrastive}.
\]

The LM objective remains the ordinary assistant-only causal loss. DAST is not
trained by replacing the answer with hidden-state supervision.

## 7. Operator dictionary

For operator arms, a router predicts mixture weights from the current state:

\[
\alpha_k=\operatorname{softmax}(W_r z_k/\tau).
\]

Each operator is low rank, \(O_j=A_jB_j\), and the predicted transition is:

\[
\hat z_{k+1}=T_{\alpha_k}(z_k)=
\operatorname{norm}\left(z_k+\sum_j\alpha_{k,j}(z_kA_j)B_j\right).
\]

Losses:

- transition target: \(1-\hat z_{k+1}^{\top}s_{k+1}^*\);
- direct-state consistency: \(1-\hat z_{k+1}^{\top}\operatorname{sg}(z_{k+1})\);
- operator diversity: squared off-diagonal cosine similarity of flattened operators;
- bounded router load diagnostic/regularizer.

The next direct state is stop-gradient in the consistency term so both sides cannot
move together into a shortcut. State alignment remains the anchor.

## 8. Causal feedback phase

A readable state can be epiphenomenal. DAST therefore contains an optional,
separately gated feedback mechanism. At checkpoint \(k\), a decoded state residual
is injected into the last prompt token:

\[
H^{(d_k)}_{t_{last}}\leftarrow H^{(d_k)}_{t_{last}}+
\gamma_k D(z_k).
\]

The gate starts near zero. During intervention evaluation, \(z_k\) can be negated,
zeroed, replaced, or swapped across examples. A valid causal result requires a
specific downstream change, not merely any logit perturbation.

Feedback is blocked with gradient checkpointing and cached generation in V1. The
offline smoke proves the hook and intervention path execute; it does not establish
semantic directionality on a pretrained model.

## 9. Diagnostics

For canonical targets, DAST computes the cross-stage similarity matrix:

\[
C_{ij}=z_i^\top s_j^*.
\]

Reported diagnostics include diagonal cosine, final-state cosine, diagonal-minus-
off-diagonal order margin, expected-checkpoint top-1 accuracy, and monotonic decoded
progress. These are mechanism diagnostics, not substitutes for generated-answer
accuracy.

## 10. What would falsify the hypothesis

The V1 depth-alignment hypothesis is not supported when any of the following holds
across frozen seeds:

- ordered fails to beat shuffled and reverse on the preregistered primary metric;
- final-state-repeated matches ordered, indicating generic deep supervision;
- ordered states are decodable but generated capability does not improve or degrades;
- counterfactual state interventions do not produce directionally appropriate outcomes;
- gains disappear under matched token/update/problem exposure;
- improvements occur only in in-domain templates and not held-out generators or deeper traces.

A negative result is retained as a closed tested form, not deleted or reframed.
