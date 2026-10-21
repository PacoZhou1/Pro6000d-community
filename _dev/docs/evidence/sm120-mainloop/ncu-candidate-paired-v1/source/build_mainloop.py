#!/usr/bin/env python3
"""Emit or build isolated SM120 M64N256 blockwise-FP8 warp-layout variants.

Emit-only is CPU-only. Compilation and all GPU execution are controlled by the
parent experiment. Sources are pinned to fixed v6 and fixed CUTLASS hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEDULER_BUILDER = HERE.parent / "sm120-scheduler" / "build_scheduler.py"
SCHEDULER_BUILDER_SHA = "4e8e14d5cf637c0998de3571098a3e924fb20b9ab369b974fab60b1ddd37a4b9"
BLOCKWISE_BUILDER_REL = "cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl"
BLOCKWISE_BUILDER_SHA = "40409c39fbbc5f023e8030472efab2a7b94baf41109eaa59fb009e52ce0d6509"
VARIANTS = {
    "control42": (4, 2, 32, 4),
    "warp24": (2, 4, 32, 4),  # Retained failed base; never reinterpret its source.
    "warp24x2": (2, 4, 32, 2),
    "warp24n64": (2, 4, 64, 4),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_builder():
    if sha(SCHEDULER_BUILDER) != SCHEDULER_BUILDER_SHA:
        raise ValueError("pinned scheduler builder changed")
    spec = importlib.util.spec_from_file_location("pinned_scheduler_builder", SCHEDULER_BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_atom_layout(source: str, variant: str, base) -> str:
    atom_m, atom_n, perm_n, b_copy_width = VARIANTS[variant]
    old = ("  static constexpr bool IsCooperative = !cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag>;\n"
           "  using AtomLayoutMNK = cute::conditional_t<IsCooperative,\n"
           "      Layout<Shape<_4,_2,_1>>, Layout<Shape<_2,_2,_1>>>;\n")
    new = ("  static constexpr bool IsCooperative = !cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag>;\n"
           "  // Pinned experiment: leave every tile except Cooperative M64N256K128 unchanged.\n"
           "  static constexpr bool IsMainloopExperimentTile = IsCooperative &&\n"
           "      size<0>(TileShape_MNK{}) == 64 && size<1>(TileShape_MNK{}) == 256 &&\n"
           "      size<2>(TileShape_MNK{}) == 128;\n"
           "  using AtomLayoutMNK = cute::conditional_t<IsMainloopExperimentTile,\n"
           f"      Layout<Shape<_{atom_m},_{atom_n},_1>>,\n"
           "      cute::conditional_t<IsCooperative,\n"
           "          Layout<Shape<_4,_2,_1>>, Layout<Shape<_2,_2,_1>>>>;\n")
    source = base.replace(source, old, new)
    if variant == "warp24x2":
        # Only the selected Cooperative M64N256K128 uses the narrower B copy.
        old_b = "  using SmemCopyAtomB = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseF8f6f4>()), SmemAllocTypeB>;"
        new_b = ("  using SmemCopyAtomB = Copy_Atom<cute::conditional_t<IsMainloopExperimentTile,\n"
                 "      cute::SM75_U32x2_LDSM_N,\n"
                 "      decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseF8f6f4>())>, SmemAllocTypeB>;")
        source = base.replace(source, old_b, new_b)
    elif variant == "warp24n64":
        old_n = "  using PermTileN = decltype(cute::min(size<1>(TileShape_MNK{}),  _32{}));"
        new_n = ("  using PermTileN = cute::conditional_t<\n"
                 "      !cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag> &&\n"
                 "      size<0>(TileShape_MNK{}) == 64 && size<1>(TileShape_MNK{}) == 256 &&\n"
                 "      size<2>(TileShape_MNK{}) == 128,\n"
                 "      decltype(cute::min(size<1>(TileShape_MNK{}), _64{})),\n"
                 "      decltype(cute::min(size<1>(TileShape_MNK{}), _32{}))>;")
        source = base.replace(source, old_n, new_n)
    return source


def patch_visitor(source: str, variant: str, revision: str, base) -> str:
    if variant == "control42":
        return source
    # The validated 4x2 v6 shadow visitor chose 2 warps per SF128. CUTLASS's
    # existing 4-warp branch states its 2x4 layout assumption explicitly.
    # This is a separate hashed layout correction, not a math relaxation.
    source = base.replace(source,
        "constexpr int WarpsPerSF = SFVecSize == 128 ? 2 : SFVecSize / 16;",
        "constexpr int WarpsPerSF = SFVecSize == 128 ? 4 : SFVecSize / 16;")
    if revision == "v3":
        # In the 2x4 ownership map, an SF128 row uses 0/2/4/6 or 1/3/5/7.
        # After the four-warp amax reduction, only warp 0 or 1 respectively
        # may write the scale. v2 kept warp<4 and is retained as failed/unsafe
        # evidence rather than silently rewritten.
        old = ("        if constexpr (IsInterWarpReductionNeeded) {\n"
               "          write_sf &= warp_idx < NumWarpsPerWarpGroup;\n"
               "        }")
        new = ("        if constexpr (IsInterWarpReductionNeeded) {\n"
               "          if constexpr (SFVecSize == 128 && WarpsPerSF == 4) {\n"
               "            // 2x4 SF128 owns {0,2,4,6} or {1,3,5,7}; one writer per group.\n"
               "            write_sf &= warp_idx < 2;\n"
               "          } else {\n"
               "            write_sf &= warp_idx < NumWarpsPerWarpGroup;\n"
               "          }\n"
               "        }")
        source = base.replace(source, old, new)
    return source


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=(*VARIANTS, "warp18"), required=True)
    p.add_argument("--visitor-revision", choices=("v2", "v3"), default="v2",
                   help="v3 adds one-writer SF128 store for 2x4; v2 retains recorded behavior")
    p.add_argument("--v6-source-dir", type=Path, required=True)
    p.add_argument("--cutlass-include", type=Path, required=True)
    p.add_argument("--cutlass-util-include", type=Path)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    p.add_argument("--emit-only", action="store_true")
    a = p.parse_args()
    if a.variant == "warp18":
        p.error("warp18 deferred: SF128 visitor supports only 1/2/4-warp reduction; an 8-warp ownership proof is required")
    if a.visitor_revision == "v3" and a.variant in ("control42", "warp24"):
        p.error("v3 unique-writer experiment applies only to compilable 2x4 variants warp24x2/warp24n64")

    base = base_builder()
    source, inc, build = (x.resolve() for x in (a.v6_source_dir, a.cutlass_include, a.build_dir))
    if build == source or source in build.parents or build == inc or inc in build.parents:
        p.error("build directory must not modify pinned source/include directories")
    base.check_files(source, base.V6_FILES)
    base.check_files(inc, base.CUTLASS_FILES)
    if sha(inc / BLOCKWISE_BUILDER_REL) != BLOCKWISE_BUILDER_SHA:
        raise ValueError("fixed SM120 blockwise builder SHA mismatch")
    manifest = json.loads((source / "source-manifest.json").read_text())
    if any(manifest.get(k) != v for k, v in base.V6_FILES.items()):
        raise ValueError("v6 source manifest differs from pinned files")
    if manifest.get("libgateup_quant_fp8.so") != base.V6_LIBRARY_SHA256:
        raise ValueError("v6 source library lineage mismatch")
    util = a.cutlass_util_include.resolve() if a.cutlass_util_include else inc.parent / "tools/util/include"
    if not (util / "cutlass/util/packed_stride.hpp").is_file():
        raise ValueError("CUTLASS tools/util/include missing")

    suffix = "" if a.visitor_revision == "v2" else "_v3"
    namespace = f"pro6000d_mainloop_{a.variant}{suffix}"
    atom_m, atom_n, perm_n, b_copy_width = VARIANTS[a.variant]
    build.mkdir(parents=True, exist_ok=True)
    generated = []
    for name in base.V6_FILES:
        if name.startswith("include/"):
            destination = build / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
            generated.append(name)
    builder_shadow = build / "include" / BLOCKWISE_BUILDER_REL
    builder_shadow.parent.mkdir(parents=True, exist_ok=True)
    builder_shadow.write_text(patch_atom_layout((inc / BLOCKWISE_BUILDER_REL).read_text(), a.variant, base))
    generated.append("include/" + BLOCKWISE_BUILDER_REL)
    visitor_shadow = build / "include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp"
    visitor_shadow.write_text(patch_visitor(visitor_shadow.read_text(), a.variant, a.visitor_revision, base))

    visitor = base.private_visitor_types((source / "paired_gateup_visitor.cuh").read_text())
    dispatch = base.dispatch_patch((source / "standalone_dispatch.cuh").read_text())
    adapter = base.private_adapter((source / "standalone_adapter.cuh").read_text())
    # Names in cutlass::epilogue::fusion live outside the private vLLM namespace.
    # Give each library its own visitor C++ identifiers as well.
    visitor_names = {
        "SchedulerPairedGateUpFp8V1": f"MainloopPairedGateUpFp8_{a.variant}{suffix}",
        "SchedulerPairedGateUpStoreV1": f"MainloopPairedGateUpStore_{a.variant}{suffix}",
    }
    for old, new in visitor_names.items():
        visitor = visitor.replace(old, new)
        dispatch = dispatch.replace(old, new)
    for name, content in (("paired_gateup_visitor.cuh", visitor),
                          ("standalone_dispatch.cuh", dispatch),
                          ("standalone_adapter.cuh", adapter)):
        (build / name).write_text(content.replace(base.PRIVATE_NAMESPACE, namespace))
        generated.append(name)
    original_pack = base.pack_body((source / "gateup_quant_driver.cu").read_text())
    pack_name = f"mainloop_pack_gateup_fp8_{a.variant}{suffix}"
    pack = base.replace(original_pack, "pack_gateup_fp8", pack_name)
    (build / "mainloop_pack.cuh").write_text("#pragma once\n" + pack)
    generated.append("mainloop_pack.cuh")
    for name in ("mainloop_driver.cu", "mainloop_abi.h"):
        shutil.copyfile(HERE / name, build / name)
        generated.append(name)

    output = build / f"libsm120_mainloop_{a.variant}{suffix}.so"
    cmd = [str(a.nvcc), "-std=c++17", "-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda",
           "-arch=sm_120a", "-lineinfo", "--resource-usage",
           "-Xcompiler=-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden,-fno-gnu-unique", "-shared",
           f"-DPRO6000D_MAINLOOP_NAMESPACE={namespace}",
           f'-DPRO6000D_MAINLOOP_VARIANT="{a.variant}{suffix}"',
           f"-DPRO6000D_MAINLOOP_ATOM_M={atom_m}", f"-DPRO6000D_MAINLOOP_ATOM_N={atom_n}",
           f"-DPRO6000D_MAINLOOP_PERM_N={perm_n}",
           f"-DPRO6000D_MAINLOOP_B_COPY_WIDTH={b_copy_width}",
           f"-DPRO6000D_MAINLOOP_PACK_KERNEL={pack_name}",
           "-I", str(build / "include"), "-I", str(build), "-I", str(inc), "-I", str(util),
           str(build / "mainloop_driver.cu"), "-o", str(output), "-lcudart"]
    report = {
        "scope": "isolated M64N256K128 mainloop warp-layout probe; no GPU work during build",
        "variant": a.variant, "visitor_revision": a.visitor_revision,
        "abi_version": 1, "library": str(output), "private_namespace": namespace,
        "mma_atom_layout_mnk": [atom_m, atom_n, 1], "mma_permutation_n": perm_n,
        "m64_smem_copy_atom_b": f"SM75_U32x{b_copy_width}_LDSM_N",
        "m64_smem_copy_b_bytes_per_thread": b_copy_width * 4,
        "patch_target": "Cooperative M64N256K128 only; official M128N128 remains original",
        "visitor_sf128_warps_per_sf": 2 if a.variant == "control42" else 4,
        "visitor_sf128_unique_scale_writer": a.visitor_revision == "v3",
        "visitor_sf128_writer_rule": "warp_idx<2 for Cooperative SF128/4-warp" if a.visitor_revision == "v3" else "v2 warp_idx<4",
        "visitor_patch_scope": "SF128 warp ownership only; other conversion/scale math unchanged",
        "v6_library_lineage_sha256": base.V6_LIBRARY_SHA256, "v6_inputs": base.V6_FILES,
        "fixed_cutlass_inputs": base.CUTLASS_FILES,
        "fixed_blockwise_builder_sha256": BLOCKWISE_BUILDER_SHA,
        "cutlass_version_header_sha256": sha(inc / "cutlass/version.h"),
        "original_pack_body_sha256": hashlib.sha256(original_pack.encode()).hexdigest(),
        "private_pack_kernel_name": pack_name,
        "private_visitor_type_names": visitor_names,
        "host_isolation": {"namespace": namespace, "c3x_state_linkage": "static thread_local",
                           "elf_visibility": "hidden except ABI C exports", "gnu_unique": "disabled"},
        "scheduler_builder_dependency_sha256": SCHEDULER_BUILDER_SHA,
        "build_mainloop_sha256": sha(Path(__file__)),
        "generated_sources": {name: sha(build / name) for name in generated},
        "command": cmd, "source_dir": str(source), "cutlass_include": str(inc),
        "build_dir": str(build), "status": "sources_emitted",
        "numerical_gate": "not_run", "graph_gate": "not_run",
    }
    report_path = build / "build-manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not a.emit_only:
        version = subprocess.run([str(a.nvcc), "--version"], text=True, capture_output=True, check=True)
        report["nvcc_version"] = version.stdout
        process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (build / "compile.log").write_text(process.stdout)
        report.update({"compile_exit_code": process.returncode,
                       "compile_log_sha256": sha(build / "compile.log"),
                       "status": "compiled" if process.returncode == 0 else "compile_failed"})
        if process.returncode == 0:
            report["library_sha256"] = sha(output)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if process.returncode:
            print(process.stdout)
            print(json.dumps(report, sort_keys=True))
            return process.returncode
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
