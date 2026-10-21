# CHANGES (host-side scripts / patch), task D run

- Patch (d2_patch/*.diff) was NOT changed. prep_src.sh had completed (pristine SHA256 -c OK; patched/{speculator.py,draft_vocab.py} present).
- A smoke run (runs/smoke-32k) existed already from the interrupted agent; it passed (draft-vocab enabled 32768 rows, graphs captured, 4 requests fine). Kept as is.
- arm.py: added client kind `tf` (runs bin/tf_client.py) for the teacher-forcing diagnostic. No other change. (scripts/arm.py is the new version.)
- New scripts: tf_client.py (teacher forcing on baseline), driver.sh / driver2.sh / driver3.sh (sequencers; one ssh call each).
- d0_lm_head_steps.py used with --min-us 100 (default 1000 would hide the ~0.2 ms trimmed head).
- return_token_ids works on vLLM 0.31 chat completions (token_ids present in all 4 arms x 2 passes), so equality is token-by-token.
- No bug fix to the patch was made; no failed run to keep except `tf-base.incomplete-*` dir created by driver3's mkdir (empty, harmless).
