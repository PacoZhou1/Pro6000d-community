#!/usr/bin/env python3
"""Emit/compile an isolated scheduler probe from exact pinned v6 sources.

No CUDA initialization or GPU launch occurs here. Compilation is root-owned.
Three GEMMs share one library; only raster and swizzle are runtime controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

V6_LIBRARY_SHA256 = "e4039ddc95fa3b56f02743357b7a7193156dc500c5d8804bb77ebd3af85450d6"
V6_FILES = {
    "gateup_quant_driver.cu": "b666d34829cc96d84dc4a8879493a4e656756c41d40d1855f7007f47e7a6835d",
    "standalone_adapter.cuh": "77847a4cb6a6af766f1452b7c67bc9936798112ff18900aad05c9d3803daaaa7",
    "paired_gateup_visitor.cuh": "7b268735efcadcf56d4d73c4a40a0655f4e1ec060a7028139f85bd887579119a",
    "standalone_dispatch.cuh": "0928952e52cdc4569edcf8c0690f5604f2449496f3a53d13a6cfd284091c6eed",
    "include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp": "2920355ce886064701479b567dfec56a778042ad3cdd854b677abd7e7721c303",
    "include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp": "8c2f05d21809638a161b2d14408f61f99acf2bb86efc5a3578e658645e892135",
    "include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp": "ed49b7fbc2c212c1eaddae0dcb89ffbd24e895603ce83e9fb78d4b1aa511f269",
}
# Exact fixed-image scheduler and original headers, not current upstream.
CUTLASS_FILES = {
    "cutlass/gemm/kernel/tile_scheduler_params.h": "ef48a12e8920183e88259d0b685279c2232fc2fb12c4fb4db7e8d0fbfdc019e9",
    "cutlass/gemm/kernel/sm90_tile_scheduler.hpp": "12e80e54672617d745f36c61e8e5f6534a9a93a67afd839a53c80301ddedce0a",
    "cutlass/gemm/kernel/static_tile_scheduler.hpp": "3cf17407654833666b3be9f7e807f299779ef5f4492bf844d0da180409f2f5f6",
    "cutlass/gemm/kernel/sm100_tile_scheduler.hpp": "54ebcaee08d4fc0663169f97c7fa665cec90d29ce5d01336c2c714f2a911b010",
    "cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp": "88bfdb527044d520c98fead4f0d1d59c051950642722b5344ad656e679287bd8",
    "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp": "d5eb1277d56571ab2dd7b906f857d0bcfca227e2ec4ffcef4440937e9466b777",
    "cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp": "1a0aa252dfd0883862dcf94a7afb1af0d2aad7d002cda3041eb28e4861b36266",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(base: Path, expected: dict[str, str]) -> None:
    for name, wanted in expected.items():
        actual = digest(base / name)
        if actual != wanted:
            raise ValueError(f"SHA mismatch {base / name}: expected {wanted}, found {actual}")


def replace(text: str, old: str, new: str, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise ValueError(f"expected {count} exact anchors, found {actual}: {old[:100]!r}")
    return text.replace(old, new)


def dispatch_patch(text: str) -> str:
    text = replace(
        text,
        "  class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto>\n",
        "  class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto,\n"
        "  class StageCount_ = void>\n",
    )
    old = "cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>"
    new = "conditional_t<cute::is_void_v<StageCount_>, " + old + ", StageCount_>"
    text = replace(text, old, new, count=2)
    anchor = "template <typename OutType>\nstruct sm120_blockwise_fp8_config_pingpong {"
    ordinary = """// Ordinary BF16 output, same exact CollectiveMainloop type as v6 fusion.
// EpilogueTileAuto retains a legal ordinary store, whose storage is reported.
struct sm120_scheduler_config_paired_bf16 {
  using Fused = typename sm120_blockwise_fp8_config_paired_quant<cutlass::float_e4m3_t>::Gemm;
  using MatchedStages = cutlass::gemm::collective::StageCount<Fused::CollectiveMainloop::DispatchPolicy::Stages>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      cutlass::bfloat16_t, 1, 128, 128, Shape<_64, _256, _128>, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120,
      false, void, cutlass::epilogue::collective::EpilogueTileAuto, MatchedStages>;
};

"""
    text = replace(text, anchor, ordinary + anchor)
    text = replace(
        text,
        "int max_swizzle_size, void* native_scales = nullptr) {",
        "int max_swizzle_size, void* native_scales = nullptr, int raster_order = 0) {",
    )
    text = replace(
        text,
        "  scheduler.max_swizzle_size = max_swizzle_size;",
        "  scheduler.max_swizzle_size = max_swizzle_size;\n"
        "  using Raster = decltype(scheduler.raster_order);\n"
        "  switch (raster_order) {\n"
        "    case 0: scheduler.raster_order = Raster::Heuristic; break;\n"
        "    case 1: scheduler.raster_order = Raster::AlongM; break;\n"
        "    case 2: scheduler.raster_order = Raster::AlongN; break;\n"
        "    default: throw std::runtime_error(\"unsupported raster\");\n"
        "  }",
    )
    text = replace(
        text,
        "  // CTA rasterization: max_swizzle_size > 1 groups nearby M/N tiles in the\n"
        "  // persistent scheduler's raster, improving the temporal locality of the\n"
        "  // shared weight (B) tiles in the L2. Bit-identical to the default order\n"
        "  // (each output tile's K-reduction is unchanged).",
        "  // Probe controls change scheduler arguments only. Every configuration\n"
        "  // remains subject to full-bit numerical and changed-input Graph gates.",
    )
    return text


def pack_body(driver: str) -> str:
    start = driver.index("__global__ void pack_gateup_fp8(")
    end = driver.index("#ifdef PRO6000D_DIAGNOSTIC_IDENTITY_VISITOR", start)
    return driver[start:end]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v6-source-dir", required=True, type=Path)
    p.add_argument("--cutlass-include", required=True, type=Path)
    p.add_argument("--cutlass-util-include", type=Path)
    p.add_argument("--build-dir", required=True, type=Path)
    p.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    p.add_argument("--emit-only", action="store_true", help="host-only source generation, no compiler")
    a = p.parse_args()
    own = Path(__file__).resolve().parent
    source, inc, build = (x.resolve() for x in (a.v6_source_dir, a.cutlass_include, a.build_dir))
    if build == source or source in build.parents or build == inc or inc in build.parents:
        p.error("build directory must not modify pinned source/include directories")
    check_files(source, V6_FILES)
    check_files(inc, CUTLASS_FILES)
    manifest = json.loads((source / "source-manifest.json").read_text())
    if any(manifest.get(k) != v for k, v in V6_FILES.items()):
        raise ValueError("v6 source manifest differs from pinned files")
    if manifest.get("libgateup_quant_fp8.so") != V6_LIBRARY_SHA256:
        raise ValueError("v6 source manifest library lineage mismatch")
    util = a.cutlass_util_include.resolve() if a.cutlass_util_include else inc.parent / "tools/util/include"
    if not (util / "cutlass/util/packed_stride.hpp").is_file():
        raise ValueError("CUTLASS tools/util/include missing")
    build.mkdir(parents=True, exist_ok=True)
    generated = []
    for name in V6_FILES:
        if not name.startswith("include/") and name != "paired_gateup_visitor.cuh":
            continue
        destination = build / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        generated.append(name)
    shutil.copyfile(source / "standalone_adapter.cuh", build / "standalone_adapter.cuh")
    (build / "standalone_dispatch.cuh").write_text(dispatch_patch((source / "standalone_dispatch.cuh").read_text()))
    extracted = pack_body((source / "gateup_quant_driver.cu").read_text())
    (build / "scheduler_pack.cuh").write_text("#pragma once\n" + extracted)
    for name in ("scheduler_driver.cu", "scheduler_abi.h"):
        shutil.copyfile(own / name, build / name)
    generated += ["standalone_adapter.cuh", "standalone_dispatch.cuh", "scheduler_pack.cuh",
                  "scheduler_driver.cu", "scheduler_abi.h"]
    output = build / "libsm120_scheduler.so"
    cmd = [str(a.nvcc), "-std=c++17", "-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda",
           "-arch=sm_120a", "-lineinfo", "--resource-usage", "-Xcompiler=-fPIC", "-shared",
           "-I", str(build / "include"), "-I", str(build), "-I", str(inc), "-I", str(util),
           str(build / "scheduler_driver.cu"), "-o", str(output), "-lcudart"]
    report = json.loads((own / "manifest-template.json").read_text())
    report.update({"v6_library_lineage_sha256": V6_LIBRARY_SHA256, "v6_inputs": V6_FILES,
                   "fixed_cutlass_inputs": CUTLASS_FILES,
                   "cutlass_version_header_sha256": digest(inc / "cutlass/version.h"),
                   "pack_body_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
                   "build_scheduler_sha256": digest(Path(__file__)),
                   "generated_sources": {name: digest(build / name) for name in generated},
                   "command": cmd, "source_dir": str(source), "cutlass_include": str(inc),
                   "build_dir": str(build), "library": str(output),
                   "status": "sources_emitted", "numerical_gate": "not_run", "graph_gate": "not_run"})
    report_path = build / "build-manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not a.emit_only:
        version = subprocess.run([str(a.nvcc), "--version"], text=True, capture_output=True, check=True)
        report["nvcc_version"] = version.stdout
        process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (build / "compile.log").write_text(process.stdout)
        report.update({"compile_exit_code": process.returncode,
                       "compile_log_sha256": digest(build / "compile.log"),
                       "status": "compiled" if process.returncode == 0 else "compile_failed"})
        if process.returncode == 0:
            report["library_sha256"] = digest(output)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if process.returncode:
            print(process.stdout)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return process.returncode
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
