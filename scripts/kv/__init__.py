"""KV-cache mechanisms that operate at inference time.

Nothing here changes model weights or parameter counts. These are runtime
policies over the cache, so they compose with any surgery or healing done to the
checkpoint itself and leave the parameter frontier in `docs/RESEARCH_PROGRAM.md`
untouched.
"""
