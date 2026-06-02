#!/usr/bin/env python3
"""
GPU keepalive: maintains ~60% utilization via duty-cycle matrix multiply.
Usage: python3 gpu_keepalive.py [--target 60] [--gpus 0,1] [--log]
"""
import argparse
import time
import signal
import sys
import threading
import subprocess

import torch

TARGET_UTIL = 60          # % utilization target
CYCLE_SEC   = 1.0         # measurement window (seconds)
MATRIX_DIM  = 4096        # matrix size for load kernel
RUNNING     = True

def sig_handler(signum, frame):
    global RUNNING
    print("\n[keepalive] Caught signal, shutting down…")
    RUNNING = False

signal.signal(signal.SIGINT,  sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def get_gpu_util(gpu_id: int) -> float:
    """Return current GPU utilization % via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits",
             f"--id={gpu_id}"],
            stderr=subprocess.DEVNULL
        )
        return float(out.decode().strip())
    except Exception:
        return 0.0


def burn_gpu(gpu_id: int, target: int, log: bool):
    """Duty-cycle loop for one GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    # Allocate matrices once (stays on GPU)
    a = torch.randn(MATRIX_DIM, MATRIX_DIM, device=device, dtype=torch.float16)
    b = torch.randn(MATRIX_DIM, MATRIX_DIM, device=device, dtype=torch.float16)

    # duty = fraction of each cycle to spend doing work
    duty = target / 100.0  # initial estimate

    while RUNNING:
        cycle_start = time.perf_counter()

        # --- active phase ---
        work_end = cycle_start + CYCLE_SEC * duty
        while time.perf_counter() < work_end and RUNNING:
            torch.mm(a, b)
        torch.cuda.synchronize(device)

        # --- idle phase ---
        idle_dur = CYCLE_SEC * (1.0 - duty)
        if idle_dur > 0:
            time.sleep(idle_dur)

        # --- measure and adjust duty ---
        util = get_gpu_util(gpu_id)
        error = target - util           # positive → need more work
        duty = max(0.05, min(0.99, duty + error * 0.02))

        if log:
            print(f"[GPU {gpu_id}] util={util:.0f}%  target={target}%  duty={duty:.2f}", flush=True)

    # clean up
    del a, b
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] stopped.")


def main():
    parser = argparse.ArgumentParser(description="GPU keepalive ~60% util")
    parser.add_argument("--target", type=int, default=TARGET_UTIL,
                        help="Target GPU utilization %% (default 60)")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU ids (default 0,1)")
    parser.add_argument("--log", action="store_true",
                        help="Print utilization every cycle")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    print(f"[keepalive] Starting on GPUs {gpu_ids}, target={args.target}%  (Ctrl-C to stop)")

    threads = []
    for gid in gpu_ids:
        t = threading.Thread(target=burn_gpu, args=(gid, args.target, args.log),
                             daemon=True, name=f"gpu-{gid}")
        t.start()
        threads.append(t)

    # Main thread just waits
    try:
        while RUNNING:
            time.sleep(0.5)
    finally:
        RUNNING = False
        for t in threads:
            t.join(timeout=5)
        print("[keepalive] All workers stopped.")


if __name__ == "__main__":
    main()
