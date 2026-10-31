"""ONNX export, INT8 post-training static quantization, ONNX Runtime inference and timing."""
import os
import time
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantFormat,
                                      QuantType)

from .data import gather
from .models import set_mc


def export_onnx(model, path, W, C):
    """Export with Monte Carlo dropout active, so S stochastic passes = S rows of one batch."""
    model.eval()
    set_mc(model, True)
    dummy = torch.zeros(4, W, C)
    torch.onnx.export(model, dummy, path, input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}}, opset_version=17,
                      dynamo=False)
    set_mc(model, False)
    return path


class _Calib(CalibrationDataReader):
    def __init__(self, data, idx, n=512, bs=64):
        sel = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
        self.batches = [gather(data, sel[i:i + bs], False) for i in range(0, len(sel), bs)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.batches):
            return None
        b = self.batches[self.i]
        self.i += 1
        return {"x": b}


def quantize_int8(fp32_path, int8_path, data, calib_idx):
    quantize_static(fp32_path, int8_path, _Calib(data, calib_idx),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=False)
    return int8_path


def session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ort_predict_mc(sess, data, idx, S, bs_windows=512):
    """Mean probability and MC std from an ONNX model (S replicas per window)."""
    means, stds = [], []
    for i in range(0, len(idx), bs_windows):
        x = gather(data, idx[i:i + bs_windows], False)
        n = len(x)
        xr = np.repeat(x, S, 0)
        z = sess.run(None, {"x": xr})[0].reshape(n, S)
        p = 1 / (1 + np.exp(-z))
        means.append(p.mean(1))
        stds.append(p.std(1))
    return np.concatenate(means), np.concatenate(stds)


def benchmark_ms(sess, W, C, S, runs=200, warmup=20):
    """Median wall time (ms) for one sensing window with S Monte Carlo replicas."""
    x = np.random.randn(S, W, C).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"x": x})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"x": x})
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def size_kb(path):
    return os.path.getsize(path) / 1024
