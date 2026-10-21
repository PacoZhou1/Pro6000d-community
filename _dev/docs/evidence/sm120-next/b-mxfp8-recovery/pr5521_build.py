#!/usr/bin/env python3
"""Emit an SM120 FP8 gate/up GEMM derivative from fixed FlashInfer PR #5521.

No GPU operation occurs unless --build is requested; --build invokes nvcc only.
The emitted kernel still requires independent GPU compilation, numerical tests,
Graph replay checks, and a same-input benchmark before any performance claim.
"""

import argparse
import errno
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source" / "flashinfer_pr5521_fc1.cu"
SOURCE_SHA256 = "e9b3c49e10b63ecd439737738e7ea0cbd713dcab9fddcc89ef1a025b221cba63"
HEADER = HERE / "pr5521_abi.h"
OPEN = "namespace h3_fc1_swiglu_gemm_fp8_sm120a {"
CLOSE = "}  // namespace h3_fc1_swiglu_gemm_fp8_sm120a"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_count(text: str, old: str, new: str, expected: int) -> str:
    actual = text.count(old)
    if actual != expected:
        raise ValueError(f"expected {expected} occurrences of {old!r}; got {actual}")
    return text.replace(old, new)


def validate_layout() -> None:
    # Host-only bijection checks. They do not establish CUDA numerical or
    # race freedom, but catch a wrong gate/up row or output tile permutation.
    g = 17408
    packed = [16 * (c // 8) + c % 8 + 8 * is_up
              for is_up in (0, 1) for c in range(g)]
    if sorted(packed) != list(range(2 * g)):
        raise ValueError("gate/up weight prepack is not a permutation")
    columns = [tile_n * 128 + warp_n * 32 + lane2 * 2 + col_group * 8 + pair
               for tile_n in range(136) for warp_n in range(4)
               for lane2 in range(4) for col_group in range(4)
               for pair in range(2)]
    if sorted(columns) != list(range(g)):
        raise ValueError("canonical output tile columns are not a permutation")


def derive() -> tuple[str, dict]:
    validate_layout()
    raw = SOURCE.read_bytes()
    if sha256(raw) != SOURCE_SHA256:
        raise ValueError("fixed PR source SHA256 mismatch")
    source = raw.decode("utf-8")
    if source.count(OPEN) != 1 or source.count(CLOSE) != 1:
        raise ValueError("FP8 namespace boundary changed")
    body = source[source.index(OPEN):source.index(CLOSE) + len(CLOSE)]
    original_accum = re.findall(r"\baccum(?:_1)?\[[^\]]+\]", body)

    # Only tile-control arithmetic changes. In particular, accum[84] and
    # accum[112] are register fragment indices and must remain unchanged.
    edits = {}
    for old, new, expected in (
        ("(load_g / 84) * 84", "(load_g / 80) * 80", 4),
        ("load_g / 84", "load_g / 80", 20),
        ("k_tile < 84", "k_tile < 80", 1),
        ("k_tile_1 < 84", "k_tile_1 < 80", 1),
        ("GROUP_M * 112", "GROUP_M * 136", 36),
    ):
        edits[old] = expected
        body = replace_count(body, old, new, expected)
    if re.search(r"load_g / 84|k_tile(?:_1)? < 84|GROUP_M \* 112", body):
        raise ValueError("old tile dimensions survived")

    # Each h_raw vector contains gate low/high and up low/high row fragments.
    # Preserve the original FP32 dequant multiplication and BF16 RN conversion;
    # remove only BF16->FP32 unpack, SiLU, and the old 14336-wide stores.
    replaced = []
    for phase in ("", "_1"):
        for row_group in range(4):
            for col_group in range(4):
                name = f"h_raw_{row_group}_{col_group}_bf16{phase}"
                start_tag = f"float h_raw_{row_group}_{col_group}_bf16_f32{phase}[8];"
                if body.count(start_tag) != 1 or body.count(f"uint32_t {name}[4];") != 1:
                    raise ValueError(f"expected one BF16 epilogue group: {name}")
                start = body.index(start_tag)
                row_lo = f"row_lo_{row_group}{phase}"
                row_hi = f"row_hi_{row_group}{phase}"
                end_tag = f"if ({row_hi} < M) {{"
                end_start = body.find(end_tag, start)
                if end_start < 0:
                    raise ValueError(f"missing high-row store: {name}")
                end_match = re.search(r"\n\s*\}", body[end_start:])
                if not end_match:
                    raise ValueError(f"unclosed high-row store: {name}")
                end = end_start + end_match.end()
                old_block = body[start:end]
                if (old_block.count("reinterpret_cast<unsigned int*>(out") != 2
                        or old_block.count("14336") != 2
                        or f"if ({row_lo} < M)" not in old_block
                        or f"y_hi_{row_group}_{col_group}_bf16{phase}[0]" not in old_block):
                    raise ValueError(f"unexpected epilogue body: {name}")
                offset = f" + {8 * col_group}" if col_group else ""
                column = (f"tile_n{phase} * 128 + warp_n{phase} * 32 "
                          f"+ (lane & 3) * 2{offset}")
                new_block = (
                    f"// PR5521-derived canonical BF16 gate/up store; original BF16 RN above.\n"
                    f"                const int canonical_col_{row_group}_{col_group}{phase} = {column};\n"
                    f"                if ({row_lo} < M) {{\n"
                    f"                    out[({row_lo} * 34816 + canonical_col_{row_group}_{col_group}{phase}) / 2] = {name}[0];\n"
                    f"                    out[({row_lo} * 34816 + 17408 + canonical_col_{row_group}_{col_group}{phase}) / 2] = {name}[2];\n"
                    f"                }}\n"
                    f"                if ({row_hi} < M) {{\n"
                    f"                    out[({row_hi} * 34816 + canonical_col_{row_group}_{col_group}{phase}) / 2] = {name}[1];\n"
                    f"                    out[({row_hi} * 34816 + 17408 + canonical_col_{row_group}_{col_group}{phase}) / 2] = {name}[3];\n"
                    f"                }}"
                )
                body = body[:start] + new_block + body[end:]
                replaced.append(name)

    # The two uncalled helper *definitions* remain source-faithful. No epilogue
    # call to either approximate exponential or reciprocal may survive.
    if (len(replaced) != 32 or "14336" in body or
            body.count("approx_exp2(") != 1 or body.count("approx_rcp(") != 1):
        raise ValueError("the SwiGLU epilogue was not fully removed")
    if original_accum != re.findall(r"\baccum(?:_1)?\[[^\]]+\]", body):
        raise ValueError("MMA register indices changed")
    if body.count("out[(") != 128:
        raise ValueError("expected four canonical writes per 32 BF16 groups")
    body = body.replace(OPEN, "namespace pro6000d_pr5521_fp8_derived {", 1)
    body = body.replace(CLOSE, "}  // namespace pro6000d_pr5521_fp8_derived", 1)
    prefix = source[:source.index(OPEN)]
    prefix = prefix[:prefix.index("namespace h3_fc1_norm_adaln_quant_fp8_sm120a {")]
    # Keep fixed copyright and CUDA headers, not the upstream norm kernel.
    prefix = prefix[:prefix.index("// MiniMax-H3 quantized fused RMSNorm")]
    generated = (prefix +
                 "// Experimental derivative of FlashInfer PR #5521 FC1 FP8 kernel.\n"
                 "// GEMM only: BF16 canonical gate/up output, K=5120, N=34816.\n"
                 "#include <cuda.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n"
                 "#include <cstdint>\n#include <math_constants.h>\n"
                 "static_assert(sizeof(CUtensorMap) == 128, \"CUDA tensor-map ABI\");\n"
                 "static_assert(alignof(CUtensorMap) >= 64, \"CUDA tensor-map alignment\");\n"
                 + body + "\n" + host_wrapper())
    return generated, {"source_sha256": SOURCE_SHA256, "epilogue_groups": replaced,
                       "tile_edit_counts": edits, "retained_accum_references": len(original_accum)}


def host_wrapper() -> str:
    return r'''
#include "pr5521_abi.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
std::mutex config_mutex;
std::vector<int> configured_devices;

int fail(const char* what) { last_error = what; return 1; }

int encode_byte_tile(CUtensorMap* descriptor, const void* base,
                     uint64_t inner_bytes, uint64_t rows,
                     uint32_t box_inner, uint32_t box_rows) {
    uint64_t global_dim[2] = {inner_bytes, rows};
    uint64_t global_strides[1] = {inner_bytes};
    uint32_t box_dim[2] = {box_inner, box_rows};
    uint32_t element_strides[2] = {1, 1};
    CUresult result = cuTensorMapEncodeTiled(
        descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base),
        global_dim, global_strides, box_dim, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return result == CUDA_SUCCESS ? 0 : fail("cuTensorMapEncodeTiled failed");
}
}  // namespace

extern "C" PR5521_ABI int recovery_pr5521_abi_version(void) { return 1; }
extern "C" PR5521_ABI const char* recovery_pr5521_last_error(void) {
    return last_error.c_str();
}

extern "C" PR5521_ABI int recovery_pr5521_launch(
    void* out, const void* a, const void* b, const float* scale_a,
    const float* scale_b, int m, int n, int k, unsigned long long stream_bits) {
    last_error.clear();
    if (!out || !a || !b || !scale_a || !scale_b ||
        (m != 2741 && m != 16160) || n != 34816 || k != 5120)
        return fail("pointer or fixed shape guard failed");
    const void* pointers[] = {out, a, b, scale_a, scale_b};
    for (const void* ptr : pointers) {
        if ((reinterpret_cast<uintptr_t>(ptr) & 15u) != 0)
            return fail("device pointer is not 16-byte aligned");
    }
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return fail(cudaGetErrorString(status));
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return fail(cudaGetErrorString(status));
    if (properties.major != 12 || properties.minor != 0)
        return fail("requires SM120");
    {
        std::lock_guard<std::mutex> lock(config_mutex);
        if (std::find(configured_devices.begin(), configured_devices.end(), device)
            == configured_devices.end()) {
            status = cudaFuncSetAttribute(
                pro6000d_pr5521_fp8_derived::kernel_h3_fc1_swiglu_gemm,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 99328);
            if (status != cudaSuccess) return fail(cudaGetErrorString(status));
            configured_devices.push_back(device);
        }
    }
    CUtensorMap a_map{}, b_map{}, unused_map{};
    if (encode_byte_tile(&a_map, a, 5120, m, 64, 128) != 0 ||
        encode_byte_tile(&b_map, b, 5120, 34816, 64, 256) != 0)
        return 1;
    const int num_m_tiles = (m + 127) / 128;
    const int total_tiles = num_m_tiles * 136;
    const int grid = std::min(total_tiles, properties.multiProcessorCount);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_bits);
    pro6000d_pr5521_fp8_derived::kernel_h3_fc1_swiglu_gemm
        <<<grid, 256, 99328, stream>>>(
            a_map, b_map, unused_map, unused_map,
            const_cast<float*>(scale_a), const_cast<float*>(scale_b),
            static_cast<unsigned int*>(out), m, num_m_tiles, total_tiles, 1.0f);
    status = cudaGetLastError();
    if (status != cudaSuccess) return fail(cudaGetErrorString(status));
    return 0;
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "build-emit")
    parser.add_argument("--build", action="store_true", help="invoke nvcc for the derived DSO")
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    generated, provenance = derive()
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = args.out / "pr5521_derived.cu"
    abi_path = args.out / "pr5521_abi.h"
    so_path = args.out / "libpr5521_derived.so"
    manifest_path = args.out / "pr5521_manifest.json"
    source_path.write_text(generated)
    shutil.copyfile(HEADER, abi_path)
    manifest = {
        "description": "experimental gate/up GEMM derived from FlashInfer PR 5521; not upstream API",
        "source": str(SOURCE), "generated": str(source_path),
        "source_sha256": provenance["source_sha256"],
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "generated_sha256": sha256(generated.encode()),
        "abi_sha256": sha256(abi_path.read_bytes()),
        "epilogue_group_count": len(provenance["epilogue_groups"]),
        "tile_edit_counts": provenance["tile_edit_counts"],
        "retained_accum_references": provenance["retained_accum_references"],
        "shapes": [[2741, 34816, 5120], [16160, 34816, 5120]],
        "output": "canonical BF16 gate/up [M,34816]",
        "timing_boundary": "ABI encodes 2 TMA descriptors per call; CUDA event timing measures launched kernel, wall clock includes descriptor setup",
        "gpu_compiled": False, "numeric_validated": False,
        "nvcc_command": None, "nvcc_exit_code": None,
        "compile_log": None, "compile_log_sha256": None,
    }
    exit_code = 0
    if args.build:
        command = [args.nvcc, "-std=c++17", "-O3", "-shared", "-Xcompiler", "-fPIC",
                   "-gencode", "arch=compute_120a,code=sm_120a",
                   "--resource-usage", "-lineinfo",
                   str(source_path), "-lcuda", "-o", str(so_path)]
        manifest["nvcc_command"] = command
        log_path = args.out / "pr5521_compile.log"
        manifest["compile_log"] = str(log_path)
        # Never let a prior successful DSO masquerade as this build's output.
        so_path.unlink(missing_ok=True)
        with log_path.open("wb") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        check=False)
                exit_code = result.returncode
            except OSError as exc:
                exit_code = 127 if exc.errno == errno.ENOENT else 1
                log.write(f"nvcc launch failed: {exc}\n".encode())
        manifest["nvcc_exit_code"] = exit_code
        manifest["compile_log_sha256"] = sha256(log_path.read_bytes())
        if exit_code == 0 and so_path.is_file():
            manifest["gpu_compiled"] = True
            manifest["library_sha256"] = sha256(so_path.read_bytes())
        elif exit_code == 0:
            exit_code = 1
            manifest["build_error"] = "nvcc returned zero without creating the DSO"
        else:
            manifest["build_error"] = "nvcc failed; see compile_log"
        manifest["build_exit_code"] = exit_code
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"generated": str(source_path), "manifest": str(manifest_path),
                      "built": manifest["gpu_compiled"], "build_exit_code": exit_code,
                      "source_sha256": SOURCE_SHA256,
                      "generated_sha256": manifest["generated_sha256"]}))
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
