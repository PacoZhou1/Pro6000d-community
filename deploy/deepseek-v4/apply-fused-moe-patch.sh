#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Applies the sole R33 host-mounted source change to a pinned B12X checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
B12X_DIR=${1:?Usage: $0 /absolute/path/to/b12x-checkout}
EXPECTED=59d51a36a942d56a9c36265855cdc7856fa7712e
ACTUAL=$(git -C "$B12X_DIR" rev-parse HEAD)
[[ "$ACTUAL" == "$EXPECTED" ]] || {
  echo "Expected B12X $EXPECTED, found $ACTUAL" >&2
  exit 2
}
cd "$B12X_DIR"
git apply --check "$ROOT/patches/deepseek-v4/fused_moe.m4-packed-r33.patch"
git apply "$ROOT/patches/deepseek-v4/fused_moe.m4-packed-r33.patch"
printf 'Applied R33 M4 packed-route selection patch to %s\n' "$B12X_DIR"
