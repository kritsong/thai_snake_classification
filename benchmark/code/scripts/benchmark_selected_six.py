"""Measure warm batch-one inference on the selected final checkpoints."""

import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_original_six_snakeclef as external
from run_original_six_grouped import atomic_json, logits_only, sha256


def forward(model, value, device):
    # Match the evaluation precision and autocast scope used by local/external
    # predictions. Timing includes host dispatch and synchronization, but the
    # normalized input tensor is already resident on the requested device.
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        return logits_only(model(value))


def measure(model, value, device, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            forward(model, value, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        values = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            forward(model, value, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            values.append((time.perf_counter_ns() - start) / 1e6)
    return {"warmup_forwards": warmup, "timed_forwards": repeats,
            "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
            "serial_images_per_second": float(1000 / np.mean(values)),
            "timing_samples_ms": values,
            "peak_torch_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else None}


def main():
    state = external.read_json(external.EXPERIMENT / "grid_progress.json")
    if state["phase"] != "complete" or not torch.cuda.is_available():
        raise ValueError("Benchmark requires completed training and the specified CUDA environment")
    output = external.EXPERIMENT / "final_analysis" / "inference_benchmark.json"
    if output.exists():
        raise FileExistsError("Existing benchmark is retained; do not silently replace timing measurements")
    labels, _, selection = external.preflight()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(external.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tensor = torch.randn(1, 3, 224, 224)
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    results = []
    for name, timm_name in external.MODELS.items():
        choice = selection["choices"][name]
        folder = external.EXPERIMENT / "images100" / name.replace("/", "-") / choice["regime"] / choice["augmentation"]
        checkpoint = folder / "final_model" / "final.pt"
        model = timm.create_model(timm_name, pretrained=False, num_classes=len(labels))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != external.read_json(folder / "final_metrics.json")["parameters"]:
            raise ValueError("Parameter count differs from trained model")
        cpu_measurement = measure(model, tensor, cpu, 5, 30)
        model.to(cuda)
        gpu_measurement = measure(model, tensor.to(cuda), cuda, 20, 100)
        row = {"model": name, "parameters": parameters, "checkpoint_sha256": sha256(checkpoint),
               "checkpoint_mib": checkpoint.stat().st_size / 1024 ** 2,
               "cpu_float32": cpu_measurement, "gpu_autocast_float16": gpu_measurement}
        results.append(row)
        print(json.dumps({"model": name, "cpu_median_ms": cpu_measurement["median_ms"],
                          "gpu_median_ms": gpu_measurement["median_ms"],
                          "gpu_p95_ms": gpu_measurement["p95_ms"],
                          "gpu_peak_mib": gpu_measurement["peak_torch_allocated_mib"]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    record = {"status": "measured_after_training_finished", "manifest_sha256": external.MANIFEST_HASH,
              "hardware": {"gpu": torch.cuda.get_device_name(), "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
                           "cpu": "12th Gen Intel Core i9-12900KF", "cpu_threads": 4, "interop_threads": 1,
                           "platform": platform.platform()},
              "software": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
              "protocol": "One fixed device-resident 1x3x224x224 tensor; eval mode; GPU float16 autocast per forward and synchronized wall timing, CPU float32 with four threads. GPU 20 warmups/100 repeats; CPU 5 warmups/30 repeats. No file decoding, resize, transfer, softmax, or model-load time. Peak GPU memory is PyTorch allocated memory including resident weights, not total display/device usage. Single-run hardware measurements, not field-device or end-to-end deployment results.",
              "models": results}
    atomic_json(output, record)


if __name__ == "__main__":
    main()
