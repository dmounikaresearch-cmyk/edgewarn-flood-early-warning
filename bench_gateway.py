"""Measure edge inference time on the real gateway (Raspberry Pi or similar).

Copy the results/models/*.onnx files to the gateway, install onnxruntime and numpy, then run:

    python3 bench_gateway.py --model student_s0_mc_int8.onnx --samples 10
    python3 bench_gateway.py --model student_s0_det_int8.onnx --samples 1

Energy per inference: read the average current I_active (A) on a USB power meter while
this script runs in --loop mode, and the idle current I_idle with nothing running.
At supply voltage V (usually 5.1 V):

    energy_mJ = (I_active - I_idle) * V * median_ms

Put the measured median_ms and P_cpu = (I_active - I_idle) * V into edgewarn/config.py
(p_cpu_w) and re-run run_experiments.py so Tables 3-4 use gateway numbers.
"""
import argparse
import platform
import time
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--samples", type=int, default=10, help="Monte Carlo passes S (1 for deterministic model)")
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--channels", type=int, default=5)
ap.add_argument("--runs", type=int, default=1000)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--loop", type=float, default=0, help="keep running for N seconds (for power metering)")
a = ap.parse_args()

so = ort.SessionOptions()
so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
x = np.random.randn(a.samples, a.window, a.channels).astype(np.float32)
for _ in range(50):
    sess.run(None, {"x": x})
ts = []
for _ in range(a.runs):
    t0 = time.perf_counter()
    sess.run(None, {"x": x})
    ts.append((time.perf_counter() - t0) * 1000)
ts = np.array(ts)
print(f"device: {platform.machine()} {platform.processor()}  onnxruntime {ort.__version__}")
print(f"model: {a.model}  S = {a.samples}  threads = {a.threads}")
print(f"latency per window: median {np.median(ts):.3f} ms, p95 {np.percentile(ts, 95):.3f} ms, "
      f"mean {ts.mean():.3f} ± {ts.std():.3f} ms over {a.runs} runs")
if a.loop > 0:
    print(f"looping for {a.loop:.0f} s - read the power meter now")
    end = time.time() + a.loop
    n = 0
    while time.time() < end:
        sess.run(None, {"x": x})
        n += 1
    print(f"{n} inferences in {a.loop:.0f} s")
