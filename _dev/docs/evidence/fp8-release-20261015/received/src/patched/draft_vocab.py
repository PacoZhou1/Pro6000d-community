# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in reduced-vocabulary lm_head for greedy draft steps (FR-Spec style).

UNTESTED design patch. Enabled only when VLLM_DRAFT_VOCAB_IDS points at a JSON
file holding a sorted list of full-vocabulary token ids (or {"ids": [...]}).
The target model's lm_head is never modified: a BF16 row-slice copy is made
once, at load time, before CUDA graph capture. Only draft steps use it; the
target's verification / bonus sampling never goes through DraftModelSpeculator
.sample_draft and keeps the full lm_head.
"""
import hashlib
import json
import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_IDS = "VLLM_DRAFT_VOCAB_IDS"
ENV_SHA = "VLLM_DRAFT_VOCAB_SHA256"  # optional: sha256 of the ids file
ENV_STRICT = "VLLM_DRAFT_VOCAB_STRICT"  # "1": raise instead of falling back


class DraftVocabHead:
    """Row-sliced copy of the (shared) target lm_head + id map back to full ids."""

    def __init__(self, ids: torch.Tensor, weight: torch.Tensor) -> None:
        self.ids = ids  # [S] int64, ascending, on device (index -> full id)
        self.weight = weight  # [S, hidden] same dtype as lm_head, contiguous
        # GPU-side counter: an in-place tensor op IS captured in a CUDA graph and
        # replays, whereas a Python int would only be bumped at capture time.
        self.used = torch.zeros((), dtype=torch.int64, device=weight.device)

    @torch.inference_mode()
    def top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [N, hidden] @ [hidden, S] -> [N, S]; argmax over subset, map to full id.
        logits = F.linear(hidden_states, self.weight)
        self.used.add_(1)
        return self.ids[logits.argmax(dim=-1)]


def maybe_build_draft_vocab_head(speculator) -> "DraftVocabHead | None":
    """Return a head, or None (counted + logged reason) -> original path."""
    path = os.environ.get(ENV_IDS)
    if not path:
        return None
    strict = os.environ.get(ENV_STRICT) == "1"
    speculator.draft_vocab_fallbacks = fb = {}

    def refuse(reason: str):
        fb[reason] = fb.get(reason, 0) + 1
        msg = f"draft-vocab disabled ({reason}); using full lm_head for drafts"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
        return None

    model = speculator.model
    cfg = speculator.speculative_config
    lm_head = getattr(model, "lm_head", None)
    lp = getattr(model, "logits_processor", None)
    if speculator.method != "mtp":
        return refuse("method_not_mtp")
    if cfg.draft_sample_method == "probabilistic":
        return refuse("probabilistic_draft_sampling")
    if getattr(speculator, "use_acceptance_estimator", False):
        return refuse("adaptive_verification_needs_full_logits")
    if getattr(speculator, "draft_watermarker", None) is not None:
        return refuse("watermarking_needs_full_logits")
    if lm_head is None or not hasattr(lm_head, "weight"):
        return refuse("no_lm_head")
    if getattr(lm_head, "tp_size", 1) != 1:
        return refuse("tp_gt_1")
    if lp is None or lp.scale != 1.0 or lp.soft_cap is not None:
        return refuse("logit_scale_or_softcap")
    if lm_head.weight.dtype != torch.bfloat16:
        return refuse(f"lm_head_dtype_{lm_head.weight.dtype}")
    if type(lm_head.quant_method).__name__ != "UnquantizedEmbeddingMethod":
        return refuse("lm_head_quantized")
    if getattr(lp, "head_dtype", None) not in (None, lm_head.weight.dtype):
        return refuse("head_dtype_override")

    raw = open(path, "rb").read()
    want = os.environ.get(ENV_SHA)
    got = hashlib.sha256(raw).hexdigest()
    if want and want != got:
        return refuse("ids_sha256_mismatch")
    obj = json.loads(raw)
    ids = obj["ids"] if isinstance(obj, dict) else obj
    vocab = model.config.vocab_size
    if (not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= vocab
            or vocab > lm_head.weight.shape[0]):
        return refuse("ids_invalid")
    # Must contain EOS / special ids or the draft can never propose them.
    eos = getattr(model.config, "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if any(e not in set(ids) for e in eos):
        return refuse("eos_not_in_subset")

    dev = lm_head.weight.device
    idx = torch.tensor(ids, dtype=torch.int64, device=dev)
    # index_select -> new contiguous BF16 buffer; original lm_head untouched.
    sub = lm_head.weight.index_select(0, idx).contiguous()
    logger.info("draft-vocab enabled: %d rows (%.1f MiB), ids sha256=%s",
                len(ids), sub.numel() * sub.element_size() / 2**20, got)
    return DraftVocabHead(idx, sub)
