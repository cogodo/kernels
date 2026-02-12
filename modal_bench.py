# modal_bench.py
# gonna use this to access GPUs from Mac
import modal

app = modal.App("cuda-kernel-bench")

# Use an official CUDA *devel* image so nvcc is available.
ctk_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",  # pick what you need
        add_python="3.11",
    )
    .entrypoint([])  # quieter entrypoint (Modal example does this)
    .apt_install("build-essential", "cmake")
    .add_local_dir(".", remote_path="/root/proj", copy=True)
    .workdir("/root/proj")
    .run_commands(
        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
        "cmake --build build -j",
    )
)

@app.function(
    gpu="A10",          # or "T4", "L4", "A100-80GB", "H100", etc.
    image=ctk_image,
    timeout=60 * 20,
)
def run_bench(iters: int = 2000):
    import subprocess
    # Your compiled benchmark should do warmup + cudaEvent timing internally
    subprocess.run(["./build/bench", f"--iters={iters}"], check=True)

@app.local_entrypoint()
def main(iters: int = 2000):
    run_bench.remote(iters)
