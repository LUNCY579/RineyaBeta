# Results policy

Only small, reviewable evidence records belong here.

The first committed result is a deterministic synthetic build-evidence manifest
whose source commit points to the implementation commit. It authorizes only the
label `INFRASTRUCTURE_PASS_ONLY`.

Real campaign outputs should be summarized here only after:

1. grouped data and campaign manifests are frozen;
2. all matched arms complete;
3. evidence payloads verify;
4. open semantic DEV per-case outputs are available;
5. the preregistered two-control and preservation rules are applied.

Do not commit local model weights, tokenizer caches, optimizer state, or sealed /
showcase data.
