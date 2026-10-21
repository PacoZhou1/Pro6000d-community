#!/bin/bash
# Host: extract pristine vLLM files from the pinned image, verify SHA, apply patch to copies.
set -euo pipefail
R=/tmp/qwen-sm120-next-d; IMG=sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
mkdir -p $R/src/pristine $R/src/patched
VP=$(docker run --rm --entrypoint python3 $IMG -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
echo "vllm path: $VP"
declare -A F=( [speculator.py]=v1/worker/gpu/spec_decode/speculator.py [logits_processor.py]=model_executor/layers/logits_processor.py
 [qwen3_5_mtp.py]=model_executor/models/qwen3_5_mtp.py [eagle_utils.py]=v1/worker/gpu/spec_decode/eagle/utils.py
 [autoregressive_speculator.py]=v1/worker/gpu/spec_decode/autoregressive/speculator.py )
for k in "${!F[@]}"; do docker run --rm --entrypoint cat $IMG $VP/${F[$k]} > $R/src/pristine/$k; done
(cd $R/src/pristine && sha256sum -c $R/inputs/d2_patch/source-sha256.txt)
# patched tree mirrors only the one dir we touch
rm -rf $R/src/tree && mkdir -p $R/src/tree/vllm/v1/worker/gpu/spec_decode
cp $R/src/pristine/speculator.py $R/src/tree/vllm/v1/worker/gpu/spec_decode/speculator.py
(cd $R/src/tree && patch -p1 < $R/inputs/d2_patch/0001-draft_vocab-new-file.diff && patch -p1 < $R/inputs/d2_patch/0002-speculator-hook.diff)
cp $R/src/tree/vllm/v1/worker/gpu/spec_decode/{speculator.py,draft_vocab.py} $R/src/patched/
sha256sum $R/src/patched/*
