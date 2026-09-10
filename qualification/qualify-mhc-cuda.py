"""Parent-scheduled CUDA qualification; never imports or loads model weights.

Run in the exact R30 image, with stable stopped and one GPU visible. Reads two
source files and AST-loads the actual B12xMHCResidual classes, using real B12X,
vLLM workspace, weak-ref op, and BreakableCUDAGraphCapture. No serving source edit.
Writes passed=false on any failure. This is kernel/lifetime QA, not benchmark.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import weakref

ORIGINAL_SHA = "6bb1eb430a30f37ef785d671eb058a6e50793c7821fa97d73a529c6a22185ccc"
PATCH_SHA = "653d3b275dbacde4c1f1db5199f8210d327fc30cc2a69efded5149fc73bbdc32"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shapes", default="64,256,8192")
    p.add_argument("--layers", type=int, default=45)
    p.add_argument("--rounds", type=int, default=3)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(f"Refusing to overwrite qualification evidence: {a.output}")
    report = dict(passed=False, patch_sha256=None, source_sha256=None,
                  cuda_tests=[], gpu_executed=False, qualification="mHC_only")
    def save():
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(report, indent=2))
    try:
        sources = [Path(a.original).read_text(), Path(a.candidate).read_text()]
        hashes = [hashlib.sha256(s.encode()).hexdigest() for s in sources]
        report.update(source_sha256=hashes[0], patch_sha256=hashes[1])
        assert hashes == [ORIGINAL_SHA, PATCH_SHA], f"Source SHA mismatch: {hashes}"
        shapes = sorted(set(map(int, a.shapes.split(","))))
        assert shapes and min(shapes) > 0 and max(shapes) <= 8192
        assert 1 <= a.layers <= 45 and a.rounds >= 3
        os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        os.environ["HOMELAB_MHC_GRAPH_WEAK_OUTPUTS"] = "1"
        import torch
        from b12x.norm import mhc
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphCapture, eager_break_during_capture)
        from vllm.utils.torch_utils import weak_ref_tensors
        from vllm.v1.worker.workspace import (
            init_workspace_manager, reset_workspace_manager,
            current_workspace_manager, collect_cuda_graph_capture_resources,
            retain_cuda_graph_capture_resource)
        assert torch.cuda.is_available() and mhc.is_supported()
        torch.cuda.set_device(a.device)
        torch.manual_seed(619823)
        device = torch.device("cuda", a.device)
        report.update(gpu=torch.cuda.get_device_name(a.device), torch=torch.__version__,
                      shapes=shapes, layers=a.layers, rounds=a.rounds,
                      binding_calls_per_graph=2*a.layers, gpu_executed=True)

        def load_class(source, label):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                       and n.name == "B12xMHCResidual")
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
                ast.alias(name="annotations")], level=0), cls], type_ignores=[])
            ns = dict(torch=torch, _require_b12x_mhc=lambda: mhc,
                      current_workspace_manager=current_workspace_manager,
                      retain_cuda_graph_capture_resource=retain_cuda_graph_capture_resource)
            exec(compile(ast.fix_missing_locations(module), label, "exec"), ns)
            return ns["B12xMHCResidual"]
        classes = [load_class(s, label) for s, label in zip(sources, ("original", "candidate"))]
        def memory():
            torch.cuda.synchronize()
            return dict(allocated=torch.cuda.memory_allocated(),
                        reserved=torch.cuda.memory_reserved(),
                        peak_allocated=torch.cuda.max_memory_allocated(),
                        free=torch.cuda.mem_get_info()[0])

        # Eager break writes to preallocated storage and retains only weak tensor
        # args through vLLM's real decorator. The next graph consumes this storage.
        @eager_break_during_capture
        def bridge(y, out):
            torch.mul(y, 0.05, out=out)

        def build(op, m, seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            rand = lambda shape, dtype: torch.randn(shape, generator=gen, device=device, dtype=dtype)
            return dict(x=rand((m,4096),torch.bfloat16)*0.1,
                        pre_fn=rand((24,4096),torch.float32)*0.002,
                        fn=rand((24,16384),torch.float32)*0.002,
                        scale=torch.full((3,),0.1,device=device,dtype=torch.float32),
                        base=torch.zeros(24,device=device,dtype=torch.float32),
                        norm=torch.ones(4096,device=device,dtype=torch.bfloat16),
                        bridges=[torch.empty((m,4096),device=device,dtype=torch.bfloat16)
                                 for _ in range(a.layers)], op=op)

        def forward(state):
            op = state["op"]
            out = op.run_pre(state["x"], state["pre_fn"], state["scale"], state["base"],
                             norm_weight=state["norm"], norm_eps=1e-6)
            for i in range(2*a.layers-1):
                r, post, comb, y = out
                if i % 2 == 0:
                    x = state["bridges"][i//2]
                    bridge(y, x)
                else:
                    x = y * 0.05
                out = op.run_post_pre(x,r,post,comb,state["fn"],state["scale"],state["base"],
                                     norm_weight=state["norm"],norm_eps=1e-6)
            return out

        def clone_outputs(out):
            return tuple(t.detach().clone() for t in out)

        def compare(got, want):
            errors=[]
            for g,w in zip(got,want):
                assert torch.isfinite(g).all().item() and torch.isfinite(w).all().item()
                # Same math/kernels: exact equality expected. Report measured error
                # even on success. Any mismatch is a gate failure, no loose mask.
                err=(g.float()-w.float()).abs().max().item()
                errors.append(err)
                assert torch.equal(g,w), f"mHC graph/eager mismatch max_abs={err}"
            return errors

        for m in shapes:
            for arm, cls in zip(("original", "candidate"),classes):
                gc.collect(); torch.cuda.empty_cache()
                free=torch.cuda.mem_get_info()[0]
                # Conservative bound includes all original retained output buffers,
                # bridges, eager oracle and spare scratch. Never adapt KV/model.
                estimated=(2*a.layers+12)*m*41040 + a.layers*m*4096*2 + 4*1024**3
                assert estimated < free*0.8, f"Budget refused: need {estimated}, free {free}"
                init_workspace_manager(device)
                op=cls(hidden_size=4096,hc_mult=4,rms_eps=1e-6,hc_eps=1e-6,sinkhorn_iters=20)
                row=dict(arm=arm,m=m,budget_bytes=estimated,before=memory(),passed=False,
                         started_unix=time.time(),phase="warmup")
                report["cuda_tests"].append(row);save()
                print(json.dumps(dict(phase="warmup",arm=arm,m=m)),flush=True)
                observation=[]
                real_bind=op._bind
                def observe(*args,**kwargs):
                    binding=real_bind(*args,**kwargs)
                    cap=BreakableCUDAGraphCapture.current()
                    if cap is not None:
                        observation.append(dict(binding_id=id(binding),
                            guards=[cap is not None,cap._capturing,torch.cuda.is_current_stream_capturing()],
                            outputs=[weakref.ref(getattr(binding,f)) for f in
                                     ("out","post_buffer","comb_buffer","y")],
                            pointers=[getattr(binding,f).data_ptr() for f in
                                      ("out","post_buffer","comb_buffer","y")],
                            partials=weakref.ref(binding.partials)))
                    return binding
                op._bind=observe
                states=[build(op,m,901),build(op,min(64,m),902)]
                # Grow workspace and compile all shapes/kernels outside capture.
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for state in states:
                        for _ in range(2):
                            warm=forward(state)
                        del warm
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                row["phase"]="capture";save()
                print(json.dumps(dict(phase="capture",arm=arm,m=m)),flush=True)
                pool=torch.cuda.graph_pool_handle()
                captures=[]
                with torch.cuda.stream(stream):
                    for state in states:
                        start=len(observation)
                        cap=BreakableCUDAGraphCapture(pool=pool)
                        with collect_cuda_graph_capture_resources() as resources, cap:
                            strong=forward(state)
                            weak=weak_ref_tensors(strong)
                            del strong
                        captures.append(dict(cap=cap,outputs=weak,resources=resources,
                                             obs=observation[start:]))
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize();gc.collect()
                row["after_capture"]=memory()
                row["captures"]=[]
                for entry in captures:
                    refs=[r for obs in entry["obs"] for r in obs["outputs"]]
                    alive=sum(r() is not None for r in refs)
                    real_guards=all(all(obs["guards"]) for obs in entry["obs"])
                    record=dict(segments=entry["cap"].num_graphs,
                                eager_breaks=entry["cap"].num_eager_breaks,
                                resources=len(entry["resources"]),bindings=len(entry["obs"]),
                                output_owners_alive=alive,output_owners_total=len(refs),
                                all_capture_guards_true=real_guards,
                                scratch_owners_alive=sum(obs["partials"]() is not None for obs in entry["obs"]))
                    row["captures"].append(record)
                    assert real_guards and record["segments"]>=2 and record["eager_breaks"]>=1
                    assert len(entry["obs"])==2*a.layers
                    assert len(entry["resources"])==len(entry["obs"]), "Unexpected extra capture resource: inspect explicitly"
                    assert record["scratch_owners_alive"]==2*a.layers
                    assert alive==(len(refs) if arm=="original" else 0), record
                    # Resources must retain the same output addresses and scratch.
                    for obs,res in zip(entry["obs"],entry["resources"]):
                        assert (id(res)==obs["binding_id"]) == (arm=="original")
                        assert obs["pointers"]==[getattr(res,f).data_ptr() for f in
                                                 ("out","post_buffer","comb_buffer","y")]
                row["replays"]=[]
                row["phase"]="replay";save()
                print(json.dumps(dict(phase="replay",arm=arm,m=m)),flush=True)
                # Large -> small -> eager -> large, with content and weight changes.
                # Each expected output is independent storage, never old entry.output.
                for turn in range(a.rounds):
                    for index in (0,1,0):
                        state,entry=states[index],captures[index]
                        torch.cuda.synchronize()
                        state["x"].fill_(0.015*(turn+1)+0.007*index)
                        state["base"].fill_(0.02*turn)
                        state["scale"].fill_(0.1+0.01*turn)
                        state["fn"].add_(0.00001*(turn+1))
                        input_copies={k:state[k].clone() for k in
                                      ("x","pre_fn","fn","scale","base","norm")}
                        eager=forward(state)
                        oracle=clone_outputs(eager)
                        # The real returned tensors remain owned through next eager
                        # call; preserve content independently and check no overwrite.
                        saved=clone_outputs(eager)
                        another=forward(state)
                        compare(eager,saved)
                        del another,eager,saved
                        torch.cuda.synchronize()
                        entry["cap"].replay()
                        torch.cuda.synchronize()
                        actual=clone_outputs(entry["outputs"])
                        errors=compare(actual,oracle)
                        for k,v in input_copies.items():
                            assert torch.equal(state[k],v), f"input {k} mutated"
                        row["replays"].append(dict(turn=turn,index=index,max_abs=errors,passed=True))
                        del actual,oracle,input_copies,v
                row["after_replay"]=memory()
                torch.cuda.synchronize()
                for entry in captures: entry["cap"].reset()
                # Drop all diagnostic loop aliases before measuring release.
                del entry,refs,obs,res,cap,resources,weak,state,captures,states,observation
                del op,real_bind,observe
                reset_workspace_manager();gc.collect();torch.cuda.empty_cache()
                row["after_release"]=memory();row["passed"]=True
                row["phase"]="complete";row["finished_unix"]=time.time();save()
                print(json.dumps(dict(phase="complete",arm=arm,m=m,passed=True)),flush=True)
        report["maximum_shape"]=max(shapes)
        report["qualification_complete"]=(max(shapes)==8192 and a.layers==45)
        report["passed"]=bool(report["qualification_complete"] and
                              all(r["passed"] for r in report["cuda_tests"]))
        if not report["passed"]:
            report["incomplete_reason"]="Small-shape/layer smoke only; full8192/45-layer gate not met"
        save()
    except BaseException as exc:
        report["error"]=str(exc);report["traceback"]=traceback.format_exc();save()
        raise


if __name__=="__main__":
    main()
