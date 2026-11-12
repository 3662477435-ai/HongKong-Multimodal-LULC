from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.flop_counter import FlopCounterMode

from .config import load_config
from .predict_experiment_models_8class import make_ours
from .train_comparison import make_dataset
from .train_comparison_8class import build_model


ALL_MODALITIES = {"s2", "s1", "highres", "nighttime", "building", "roads", "poi", "aoi", "text"}


def _inputs(sample, device):
    return tuple(
        sample[key][None].to(device)
        for key in ("s2", "s1", "urban", "roads", "osm", "input_ids", "text_mask")
    )


def _forward(model, inputs, ours: bool):
    output = model(*inputs)
    if ours:
        return output["combined_log_probs"]
    return output


def _load_model(cfg, dataset, spec, device):
    checkpoint_path = Path(spec["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if spec["kind"] == "ours":
        model = make_ours(cfg, dataset, checkpoint_path)
        ours = True
    else:
        model, multimodal = build_model(
            cfg, dataset, spec["model"], ALL_MODALITIES, pretrained=False
        )
        if not multimodal:
            raise RuntimeError("Profiling requires a full-modal comparator")
        model.load_state_dict(checkpoint["model_state"])
        ours = False
    return model.to(device).eval(), ours, checkpoint


def _profile_one(model, inputs, ours, warmup, repeats, device):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            _forward(model, inputs, ours)
        flop_count = counter.get_total_flops()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(warmup):
                _forward(model, inputs, ours)
        torch.cuda.synchronize()
        elapsed = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _forward(model, inputs, ours)
                end.record()
                end.synchronize()
                elapsed.append(start.elapsed_time(end))
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        elapsed = []
        with torch.no_grad():
            for _ in range(warmup):
                _forward(model, inputs, ours)
            for _ in range(repeats):
                started = time.perf_counter()
                _forward(model, inputs, ours)
                elapsed.append(1000.0 * (time.perf_counter() - started))
        peak_memory = float("nan")
    values = np.asarray(elapsed, dtype=float)
    return {
        "parameter_count": int(parameter_count),
        "forward_flops": int(flop_count),
        "forward_gflops": float(flop_count / 1e9),
        "latency_ms_mean": float(values.mean()),
        "latency_ms_median": float(np.median(values)),
        "latency_ms_p05": float(np.quantile(values, 0.05)),
        "latency_ms_p95": float(np.quantile(values, 0.95)),
        "throughput_patches_per_second": float(1000.0 / values.mean()),
        "throughput_megapixels_per_second": float(128 * 128 / values.mean() / 1000.0),
        "peak_gpu_memory_mb": float(peak_memory),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_release.yaml"))
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config.resolve())
    # All reported eight-class checkpoints were trained with the canonical
    # hierarchical-v2 manifest, vocabulary and train-only normalization.
    cfg["data"]["manifest_name"] = "manifest_hier_v2.csv"
    cfg["data"]["vocab_name"] = "vocab_hier_v2.json"
    cfg["data"]["normalization_name"] = "normalization_hier_v2.json"
    dataset = make_dataset(cfg, "test", False, ALL_MODALITIES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = _inputs(dataset[0], device)
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    rows = []
    for spec in specs:
        model, ours, checkpoint = _load_model(cfg, dataset, spec, device)
        row = {"name": spec["name"], "model": spec.get("model", "ours")}
        row.update(_profile_one(model, inputs, ours, args.warmup, args.repeats, device))
        row["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "model_complexity_and_inference_speed.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "input": "one full-modal 128x128 patch; batch size 1",
        "precision": "bfloat16 autocast for latency; float32 operator graph for FLOPs",
        "flop_definition": "PyTorch FlopCounterMode forward-pass floating-point operations",
        "warmup": args.warmup,
        "timed_repeats": args.repeats,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (output / "profiling_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
