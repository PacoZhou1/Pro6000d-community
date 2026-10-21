#!/usr/bin/env bash
set -uo pipefail
source /tmp/qwen-sm120-next-a/work/common_host.sh
BUILD='NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
CUOBJDUMP=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
STUBS=$(dirname "$(find /usr/local/cuda* -name libnvidia-ml.so -print -quit)")
echo "NVCC=$NVCC STUBS=$STUBS"
$NVCC --version > /results/nvcc-version.txt
build() { $NVCC -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -lineinfo $1 -Xptxas -v -o /results/mma_rate_probe /work/mma_rate_probe.cu -ldl; }
if build "-DBS_VARIANT=0" 2> /results/build-v0.log; then echo 0 > /results/bs_variant.txt
elif build "-DBS_VARIANT=1" 2> /results/build-v1.log; then echo 1 > /results/bs_variant.txt
else echo failed > /results/bs_variant.txt; exit 1; fi
$CUOBJDUMP -sass /results/mma_rate_probe > /results/mma_rate_probe.sass
grep -E "QMMA|OMMA|HMMA" /results/mma_rate_probe.sass | sed -E "s/^[^;]*;?//" | sort | uniq -c | sort -rn > /results/sass-mma-forms.txt
cat /results/sass-mma-forms.txt'
drun build nogpu --entrypoint bash "$IMAGE" -c "$BUILD" > "$OUT/build.stdout" 2> "$OUT/build.stderr"
echo rc=$?
