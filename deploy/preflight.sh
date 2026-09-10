#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
(cd "$ROOT/patches" && sha256sum -c SHA256SUMS)
MODEL_DIR=$(docker inspect glm53-r30-dma9100 --format '{{range .Mounts}}{{if eq .Destination "/models/GLM-5.3-Flash-NVFP4"}}{{.Source}}{{end}}{{end}}')
test -s "$MODEL_DIR/config.json"
nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{n++;if($1!=350)bad=1} END{exit(bad||n!=4)}'
