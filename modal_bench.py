"""
Modal-based GEMM benchmark.

Usage:
    modal run modal_bench.py               # benchmark on B200, save results.json
    python plot_bench.py                    # generate PNG graphs from results.json
    python plot_bench.py other_results.json # use a different results file
"""

import modal
import os
import json
import ctypes
import subprocess
import glob
import re

app = modal.App("mmul-bench")

# ---------------------------------------------------------------------------
# Container image: CUDA 12.8 devel (includes nvcc) + Python deps
# .cu source files are baked into the image via add_local_dir
# ---------------------------------------------------------------------------
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("torch", "apache-tvm-ffi", "flashinfer-python")
    .add_local_dir(".", "/root/kernels", ignore=lambda pth: not pth.name.endswith(".cu"))
)

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
SIZES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

N_WARMUP = 10
N_ITER = 50

# All launch functions share this C signature:
#   int gemm_launch_<name>(const float* A, const float* B, float* C,
#                          int M, int N, int K, float alpha, float beta)
LAUNCH_ARGTYPES = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # A, B, C
    ctypes.c_int, ctypes.c_int, ctypes.c_int,            # M, N, K
    ctypes.c_float, ctypes.c_float,                       # alpha, beta
]


# ---------------------------------------------------------------------------
# Benchmark class – runs on Modal with a B200 GPU
# ---------------------------------------------------------------------------
@app.cls(gpu="B200+", image=cuda_image)
class GemmBench:
    @modal.enter()
    def compile_and_load(self):
        """Compile .cu source files into libgemm.so, then auto-discover all
        exported gemm_launch_* functions."""
        cu_files = sorted(glob.glob("/root/kernels/*.cu"))
        if not cu_files:
            raise RuntimeError("No .cu files found in /root/kernels/")

        cmd = [
            "nvcc",
            "-shared",
            "-Xcompiler", "-fPIC",
            "-arch=sm_100",
            "-O3",
            "-o", "/root/libgemm.so",
        ] + cu_files
        print(f"Compiling: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Load the shared library
        self._lib = ctypes.CDLL("/root/libgemm.so")

        # Auto-discover all exported gemm_launch_* symbols
        nm_out = subprocess.check_output(
            ["nm", "-D", "--defined-only", "/root/libgemm.so"], text=True
        )
        self._kernels: dict[str, ctypes._NamedFuncPointer] = {}
        for line in nm_out.splitlines():
            match = re.search(r"\bT (gemm_launch_\w+)", line)
            if match:
                fname = match.group(1)
                fn = getattr(self._lib, fname)
                fn.restype = ctypes.c_int
                fn.argtypes = LAUNCH_ARGTYPES
                self._kernels[fname] = fn

        if not self._kernels:
            raise RuntimeError(
                "No gemm_launch_* symbols found in libgemm.so. "
                "Make sure your launch functions are declared extern \"C\"."
            )
        print(f"Discovered kernels: {', '.join(sorted(self._kernels))}")

        # Register all with TVM-FFI
        from tvm_ffi import register_global_func
        for fname, fn in self._kernels.items():
            _fn = fn  # capture for closure

            @register_global_func(f"flashinfer.{fname}")
            def _ffi_wrapper(A, B, C, M, N, K, alpha=1.0, beta=0.0, _f=_fn):
                err = _f(
                    ctypes.c_void_p(int(A.data_ptr())),
                    ctypes.c_void_p(int(B.data_ptr())),
                    ctypes.c_void_p(int(C.data_ptr())),
                    int(M), int(N), int(K),
                    ctypes.c_float(float(alpha)),
                    ctypes.c_float(float(beta)),
                )
                if err != 0:
                    raise RuntimeError(f"{fname} CUDA error {err}")

    # -- helpers -----------------------------------------------------------

    def _call_kernel(self, fn, A, B, C, M, N, K, alpha=1.0, beta=0.0):
        """Call a GEMM launch function via ctypes."""
        err = fn(
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            M, N, K,
            ctypes.c_float(alpha),
            ctypes.c_float(beta),
        )
        if err != 0:
            raise RuntimeError(f"CUDA error {err}")

    @staticmethod
    def _time_fn(fn, n_warmup, n_iter):
        """Time a GPU function using CUDA events. Returns median time in ms."""
        import torch

        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        return times[len(times) // 2]  # median

    # -- benchmark entrypoint ----------------------------------------------

    @modal.method()
    def benchmark(
        self,
        sizes=None,
        n_warmup=N_WARMUP,
        n_iter=N_ITER,
        skip_kernels: list[str] | None = None,
    ):
        """Run benchmark sweep across matrix sizes.

        Only benchmarks kernels NOT in skip_kernels (plus always re-runs
        cuBLAS). Returns a list of dicts with timing/throughput data for the
        new kernels + cuBLAS only.
        """
        import torch

        if sizes is None:
            sizes = SIZES

        skip = set(skip_kernels or [])
        run_kernels = {
            fname: fn
            for fname, fn in self._kernels.items()
            if fname.removeprefix("gemm_launch_") not in skip
        }

        if run_kernels:
            print(f"Benchmarking: {', '.join(sorted(run_kernels))}")
        else:
            print("No new kernels to benchmark.")
        if skip:
            print(f"Skipping (already in results.json): {', '.join(sorted(skip))}")

        results = []
        for M, N, K in sizes:
            print(f"\n--- M={M}, N={N}, K={K} ---")

            A = torch.randn(M, K, device="cuda", dtype=torch.float32)
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

            flops = 2.0 * M * N * K
            # Memory: read A (M*K) + read B (K*N) + read/write C (2*M*N)
            mem_bytes = (M * K + K * N + 2 * M * N) * 4  # float32

            entry = {
                "M": M, "N": N, "K": K,
                "flops": flops,
                "mem_bytes": mem_bytes,
            }

            # --- Benchmark only new kernels --------------------------------
            for fname, fn in sorted(run_kernels.items()):
                short = fname.removeprefix("gemm_launch_")
                C.zero_()
                try:
                    ms = self._time_fn(
                        lambda _f=fn: self._call_kernel(_f, A, B, C, M, N, K),
                        n_warmup, n_iter,
                    )
                    gflops = flops / (ms * 1e-3) / 1e9
                    gbps = mem_bytes / (ms * 1e-3) / 1e9
                    entry[f"{short}_ms"] = ms
                    entry[f"{short}_gflops"] = gflops
                    entry[f"{short}_gbps"] = gbps
                    print(f"  {short}: {ms:.3f} ms  |  {gflops:.1f} GFLOPS  |  {gbps:.1f} GB/s")
                except RuntimeError as e:
                    print(f"  {short}: FAILED ({e})")
                    entry[f"{short}_ms"] = None
                    entry[f"{short}_gflops"] = None
                    entry[f"{short}_gbps"] = None

            # --- cuBLAS baseline (always re-run) ---------------------------
            C_ref = torch.zeros(M, N, device="cuda", dtype=torch.float32)
            ms = self._time_fn(
                lambda: torch.mm(A, B, out=C_ref),
                n_warmup, n_iter,
            )
            gflops = flops / (ms * 1e-3) / 1e9
            gbps = mem_bytes / (ms * 1e-3) / 1e9
            entry["cublas_ms"] = ms
            entry["cublas_gflops"] = gflops
            entry["cublas_gbps"] = gbps
            print(f"  cuBLAS: {ms:.3f} ms  |  {gflops:.1f} GFLOPS  |  {gbps:.1f} GB/s")

            results.append(entry)

        return results


# ---------------------------------------------------------------------------
# Helpers for incremental results
# ---------------------------------------------------------------------------
def _results_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def _load_existing(path: str) -> list[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _existing_kernels(results: list[dict]) -> set[str]:
    """Return the set of kernel short-names already benchmarked."""
    kernels: set[str] = set()
    for key in results[0] if results else []:
        if key.endswith("_gflops") and not key.startswith("cublas"):
            kernels.add(key.removesuffix("_gflops"))
    return kernels


def _merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new kernel columns + updated cuBLAS into existing results.

    Matches entries by (M, N, K). cuBLAS columns are always overwritten
    with the fresh run.
    """
    index: dict[tuple[int, int, int], dict] = {}
    for entry in existing:
        key = (entry["M"], entry["N"], entry["K"])
        index[key] = entry

    for entry in new:
        key = (entry["M"], entry["N"], entry["K"])
        if key in index:
            index[key].update(entry)
        else:
            index[key] = entry

    # Re-compute pct_cublas for ALL kernels after merge
    merged = list(index.values())
    for entry in merged:
        cublas = entry.get("cublas_gflops")
        if not cublas:
            continue
        for k in list(entry.keys()):
            if k.endswith("_gflops") and not k.startswith("cublas"):
                short = k.removesuffix("_gflops")
                val = entry[k]
                entry[f"{short}_pct_cublas"] = (
                    val / cublas * 100 if val is not None else None
                )

    # Sort by matrix size
    merged.sort(key=lambda e: (e["M"], e["N"], e["K"]))
    return merged


# ---------------------------------------------------------------------------
# Local entrypoint – runs on your machine, calls Modal for GPU work
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    out_path = _results_path()
    existing = _load_existing(out_path)
    already = _existing_kernels(existing)

    if already:
        print(f"Existing results have kernels: {', '.join(sorted(already))}")

    bench = GemmBench()
    new_results = bench.benchmark.remote(skip_kernels=list(already))

    merged = _merge_results(existing, new_results)

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Run `python plot_bench.py` to generate graphs.")
