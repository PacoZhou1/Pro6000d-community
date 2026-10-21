#!/usr/bin/env python3
"""D1: token-frequency subsets for draft-vocab pruning (FR-Spec style).

Run:  download/venv/bin/python -I d1_build_vocab.py
Inputs (all from ModelScope, see download_manifest.json / corpus_manifest.json):
  download/tokenizer/tokenizer.json (+config.json, tokenizer_config.json)
  download/corpus/*.jsonl  ({"text": ...} per line)
Method
  * every doc is tokenized with the Qwen3.6 tokenizer (no BOS/EOS added);
  * doc i goes to the EVAL split when i % 5 == 0, else to BUILD (held-out coverage);
  * general frequency = weighted mix of per-domain normalised BUILD frequencies
    (zh 0.30, en 0.40, code 0.30) so that no domain dominates by corpus size;
  * reasoning frequency (reason_zh + reason_en BUILD) is kept separate;
  * forced ids in EVERY subset: all added/special tokens (incl. <think>, </think>,
    <tool_call>, <|im_end|> ...), EOS ids from config/generation_config, and the 256
    byte-level tokens (so any byte string can be proposed); forced ids count inside K;
  * remaining slots: highest mixed frequency; unseen fill (if the corpus has fewer
    than K distinct tokens) by ascending id (BPE ids ~ merge rank).
Outputs under outputs/: subset_<K>.json, reasoning_subset_<K>.json, coverage.json,
coverage.md, token_counts.npz.
"""
import glob, hashlib, json, os
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "download")
OUT = os.path.join(HERE, "outputs")
KS = [16384, 32768, 65536]
WEIGHTS = {"zh": 0.30, "en": 0.40, "code": 0.30}
DOMAIN = {"zh_wiki": "zh", "zh_web": "zh", "en_web": "en", "code_python": "code",
          "code_js": "code", "code_alpaca": "code", "reason_zh": "reasoning_zh",
          "reason_en": "reasoning_en"}
MAXCHARS = 20000


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    tk = Tokenizer.from_file(os.path.join(DL, "tokenizer", "tokenizer.json"))
    cfg = json.load(open(os.path.join(DL, "tokenizer", "config.json")))
    gen = json.load(open(os.path.join(DL, "tokenizer", "generation_config.json")))
    tcfg = json.load(open(os.path.join(DL, "tokenizer", "tokenizer_config.json")))
    V = cfg.get("text_config", cfg)["vocab_size"]
    # forced ids -------------------------------------------------------------
    forced, why = set(), {}
    tj = json.load(open(os.path.join(DL, "tokenizer", "tokenizer.json")))
    for t in tj.get("added_tokens", []):
        forced.add(t["id"]); why.setdefault("added_tokens", 0); why["added_tokens"] += 1
    for k, v in (tcfg.get("added_tokens_decoder") or {}).items():
        forced.add(int(k))
    eos = []
    for src in (cfg.get("text_config", cfg).get("eos_token_id"), gen.get("eos_token_id")):
        eos += [src] if isinstance(src, int) else list(src or [])
    forced.update(eos)
    vocab = tk.get_vocab()
    # byte-level base alphabet (GPT-2 style bytes_to_unicode)
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    byte_ids = [vocab[chr(c)] for c in cs if chr(c) in vocab]
    forced.update(byte_ids)
    forced = {i for i in forced if 0 <= i < V}
    think_ids = {t: vocab.get(t) for t in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")}
    print("forced", len(forced), "byte ids", len(byte_ids), "eos", sorted(set(eos)), think_ids)

    # tokenize ----------------------------------------------------------------
    counts = {}      # source -> BUILD counts
    eval_ids = {}    # source -> EVAL token array
    ndocs = {}
    for path in sorted(glob.glob(os.path.join(DL, "corpus", "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        if name not in DOMAIN:
            continue
        build = np.zeros(V, dtype=np.int64)
        ev = []
        batch, idx, nb, ne = [], 0, 0, 0
        def flush():
            nonlocal batch, nb, ne, build
            if not batch:
                return
            encs = tk.encode_batch([t for _, t in batch], add_special_tokens=False)
            for (i, _), e in zip(batch, encs):
                ids = np.asarray(e.ids, dtype=np.int32)
                if i % 5 == 0:
                    ev.append(ids); ne += 1
                else:
                    build += np.bincount(ids, minlength=V); nb += 1
            batch = []
        with open(path, encoding="utf8") as f:
            for line in f:
                t = json.loads(line)["text"][:MAXCHARS]
                if t.strip():
                    batch.append((idx, t)); idx += 1
                if len(batch) >= 512:
                    flush()
        flush()
        counts[name] = build
        eval_ids[name] = np.concatenate(ev) if ev else np.zeros(0, np.int32)
        ndocs[name] = {"build_docs": nb, "eval_docs": ne, "build_tokens": int(build.sum()),
                       "eval_tokens": int(eval_ids[name].size)}
        print(name, ndocs[name])
    np.savez_compressed(os.path.join(OUT, "token_counts.npz"), **counts)

    def dom_freq(domain):
        tot = sum(counts[n] for n in counts if DOMAIN[n] == domain)
        return tot / max(tot.sum(), 1)
    mixed = sum(w * dom_freq(d) for d, w in WEIGHTS.items() if any(DOMAIN[n] == d for n in counts))
    reasoning_tot = sum(counts[n] for n in counts if DOMAIN[n].startswith("reasoning"))
    rfreq = reasoning_tot / max(reasoning_tot.sum(), 1)

    def pick(freq, K):
        chosen = set(forced)
        order = np.lexsort((np.arange(V), -freq))  # freq desc, id asc
        for i in order:
            if len(chosen) >= K:
                break
            if freq[i] <= 0:
                break
            chosen.add(int(i))
        filler = 0
        i = 0
        while len(chosen) < K:
            if i not in chosen:
                chosen.add(i); filler += 1
            i += 1
        return sorted(chosen), filler

    def save(name, ids, filler, extra):
        p = os.path.join(OUT, name)
        obj = {"ids": ids, "size": len(ids), "vocab_size": V,
               "forced_count": len(forced), "unseen_filler": filler,
               "tokenizer_sha256": sha256(os.path.join(DL, "tokenizer", "tokenizer.json")), **extra}
        with open(p, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        return p, sha256(p)

    ev_domain = {}
    for n, a in eval_ids.items():
        ev_domain.setdefault(DOMAIN[n], []).append(a)
    ev_domain = {d: np.concatenate(v) for d, v in ev_domain.items()}
    ev_domain["zh_all"] = ev_domain["zh"]
    groups = {"zh": ev_domain.get("zh"), "en": ev_domain.get("en"), "code": ev_domain.get("code"),
              "reasoning_zh": ev_domain.get("reasoning_zh"), "reasoning_en": ev_domain.get("reasoning_en")}
    per_source_eval = {n: a for n, a in eval_ids.items()}

    def cover(ids, arr):
        if arr is None or arr.size == 0:
            return None
        m = np.zeros(V, dtype=bool); m[ids] = True
        hit = m[arr]
        # P(next 4 tokens all inside subset): what a 4-token draft chain needs
        c4 = hit[:-3] & hit[1:-2] & hit[2:-1] & hit[3:] if arr.size > 4 else hit
        return {"token_coverage": float(hit.mean()), "chain4_coverage": float(c4.mean()),
                "tokens": int(arr.size)}

    res = {"vocab_size": V, "forced_ids": len(forced), "eos_ids": sorted(set(eos)),
           "special_examples": think_ids, "weights": WEIGHTS, "docs": ndocs, "subsets": {}}
    files = {}
    for K in KS:
        ids, filler = pick(mixed, K)
        p, h = save(f"subset_{K}.json", ids, filler, {"kind": "general_mixed", "weights": WEIGHTS})
        files[f"subset_{K}.json"] = h
        r = {"sha256": h, "unseen_filler": filler, "eval_coverage_by_domain": {g: cover(ids, a) for g, a in groups.items()},
             "eval_coverage_by_source": {n: cover(ids, a) for n, a in per_source_eval.items()}}
        # in-sample (BUILD) mass for reference
        m = np.zeros(V, dtype=bool); m[ids] = True
        r["build_mass_mixed"] = float(mixed[m].sum())
        rids, rfill = pick(rfreq, K)
        rp, rh = save(f"reasoning_subset_{K}.json", rids, rfill, {"kind": "reasoning_zh_en"})
        files[f"reasoning_subset_{K}.json"] = rh
        gset, rset = set(ids), set(rids)
        r["reasoning_subset"] = {
            "sha256": rh, "unseen_filler": rfill,
            "overlap_with_general": len(gset & rset), "jaccard": len(gset & rset) / len(gset | rset),
            "fraction_of_reasoning_subset_in_general": len(gset & rset) / len(rset),
            "eval_coverage_reasoning_by_reasoning_subset": {g: cover(rids, groups[g]) for g in ("reasoning_zh", "reasoning_en")},
            "eval_coverage_general_domains_by_reasoning_subset": {g: cover(rids, groups[g]) for g in ("zh", "en", "code")},
        }
        uids = sorted(gset | rset)
        r["union_general_reasoning"] = {"size": len(uids),
            "eval_coverage_reasoning": {g: cover(uids, groups[g]) for g in ("reasoning_zh", "reasoning_en")}}
        res["subsets"][str(K)] = r
    res["files_sha256"] = files
    json.dump(res, open(os.path.join(OUT, "coverage.json"), "w"), indent=1, ensure_ascii=False)

    # markdown ------------------------------------------------------------------
    L = ["# D1 coverage (held-out docs, token coverage / 4-token chain coverage)\n",
         "| K | zh | en | code | reasoning_zh | reasoning_en |", "|---|---|---|---|---|---|"]
    for K in KS:
        c = res["subsets"][str(K)]["eval_coverage_by_domain"]
        L.append(f"| {K} | " + " | ".join(
            (f"{c[g]['token_coverage']*100:.2f}% / {c[g]['chain4_coverage']*100:.2f}%" if c[g] else "n/a")
            for g in ("zh", "en", "code", "reasoning_zh", "reasoning_en")) + " |")
    L += ["", "## Reasoning-specific subset vs general", "| K | overlap | Jaccard | reasoning subset ids inside general | reasoning_zh cov (general / reasoning-subset) | reasoning_en cov (general / reasoning-subset) |", "|---|---|---|---|---|---|"]
    for K in KS:
        r = res["subsets"][str(K)]; rs = r["reasoning_subset"]
        gz = r["eval_coverage_by_domain"]["reasoning_zh"]; ge = r["eval_coverage_by_domain"]["reasoning_en"]
        rz = rs["eval_coverage_reasoning_by_reasoning_subset"]["reasoning_zh"]; re_ = rs["eval_coverage_reasoning_by_reasoning_subset"]["reasoning_en"]
        f = lambda a, b: (f"{a['token_coverage']*100:.2f}% / {b['token_coverage']*100:.2f}%" if a and b else "n/a")
        L.append(f"| {K} | {rs['overlap_with_general']} | {rs['jaccard']:.3f} | {rs['fraction_of_reasoning_subset_in_general']*100:.1f}% | {f(gz, rz)} | {f(ge, re_)} |")
    open(os.path.join(OUT, "coverage.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
