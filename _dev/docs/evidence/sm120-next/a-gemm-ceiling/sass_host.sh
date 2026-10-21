#!/usr/bin/env bash
set -uo pipefail
source /tmp/qwen-sm120-next-a/work/common_host.sh
SASS='export PATH=$PATH:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin
/usr/local/cuda/bin/cuobjdump -sass /results/mma_rate_probe > /results/mma_rate_probe.sass
grep -E "Function :" /results/mma_rate_probe.sass > /results/sass-functions.txt
grep -E "QMMA|OMMA|HMMA" /results/mma_rate_probe.sass > /results/sass-mma-lines.txt
sed -E "s/^[^;]*;?//; s/ +/ /g" /results/sass-mma-lines.txt | sed -E "s/R[0-9]+/R#/g; s/0x[0-9a-f]+/0x#/g" | sort | uniq -c | sort -rn > /results/sass-mma-forms.txt
cat /results/sass-functions.txt /results/sass-mma-forms.txt'
drun sass nogpu --entrypoint bash "$IMAGE" -c "$SASS" > "$OUT/sass.stdout" 2> "$OUT/sass.stderr"
cat "$OUT/sass.stdout" "$OUT/sass.stderr"
