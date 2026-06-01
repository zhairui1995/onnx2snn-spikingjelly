from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from spikingjelly.activation_based import functional

from .core import ConversionArtifacts, ConversionConfig
from .loader import load_onnx_graph
from .model import build_ann_model, build_snn_model, build_snn_surrogate_ann_model
from .patterns import analyze_patterns, pattern_report


def convert_onnx_to_snn(
    onnx_path: str | Path,
    output_dir: str | Path,
    config: ConversionConfig | dict[str, Any] | None = None,
    calibration_loader: Iterable | None = None,
    eval_loader: Iterable | None = None,
) -> ConversionArtifacts:
    """Convert a classification CNN/ResNet-style ONNX model to ANN and SNN artifacts."""

    cfg = ConversionConfig.from_user_config(config)
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_onnx_graph(str(onnx_path))
    pattern_groups = analyze_patterns(graph)
    ann_model = build_ann_model(graph).to(cfg.device).eval()
    calibration_model = build_snn_surrogate_ann_model(
        ann_model,
        replace_maxpool_with_avgpool=cfg.replace_maxpool_with_avgpool_in_snn,
    ).to(cfg.device).eval()

    calibration_batches, synthetic_used = _materialize_calibration_batches(
        calibration_loader, graph, cfg
    )
    calibration_stats = _calibrate_relu_scales(calibration_model, calibration_batches, cfg)
    snn_model = build_snn_model(
        ann_model,
        calibration_stats["relu_scales"],
        replace_maxpool_with_avgpool=cfg.replace_maxpool_with_avgpool_in_snn,
    )
    snn_model = snn_model.to(cfg.device).eval()

    report: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "supported": True,
        "opset_imports": graph.opset_imports,
        "input_names": graph.input_names,
        "output_names": graph.output_names,
        "node_count": len(graph.nodes),
        "op_counts": _op_counts(graph.nodes),
        "pattern_groups": pattern_report(pattern_groups),
        "synthetic_calibration_used": synthetic_used,
        "calibration": calibration_stats,
        "snn_graph_transform": {
            "replace_maxpool_with_avgpool": cfg.replace_maxpool_with_avgpool_in_snn,
            "maxpool_to_avgpool_count": sum(
                1 for node in graph.nodes if node.op_type == "MaxPool"
            )
            if cfg.replace_maxpool_with_avgpool_in_snn
            else 0,
        },
        "onnx_vs_ann": _compare_onnxruntime(onnx_path, ann_model, calibration_batches, cfg),
    }
    if eval_loader is not None:
        report["evaluation"] = _evaluate_models(
            ann_model,
            snn_model,
            eval_loader,
            cfg,
            surrogate_ann_model=calibration_model
            if cfg.replace_maxpool_with_avgpool_in_snn
            else None,
        )

    _write_artifacts(output_dir, ann_model, snn_model, cfg, calibration_stats, report)
    return ConversionArtifacts(
        ann_model=ann_model,
        snn_model=snn_model,
        config=cfg,
        calibration_stats=calibration_stats,
        report=report,
        output_dir=output_dir,
    )


def _materialize_calibration_batches(
    loader: Iterable | None, graph, cfg: ConversionConfig
) -> tuple[list[torch.Tensor], bool]:
    if loader is not None:
        batches = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            batches.append(x.detach().to(cfg.device).float())
        if batches:
            return batches, False

    shape = cfg.input_shape or _infer_input_shape(graph)
    if shape is None:
        raise ValueError(
            "No calibration_loader was provided and input_shape could not be inferred"
        )
    shape = tuple(cfg.synthetic_batch_size if idx == 0 else int(dim) for idx, dim in enumerate(shape))
    generator = torch.Generator(device="cpu").manual_seed(0)
    batches = [
        torch.rand(shape, generator=generator).to(cfg.device)
        for _ in range(max(int(cfg.synthetic_calibration_batches), 1))
    ]
    return batches, True


def _infer_input_shape(graph) -> tuple[int, ...] | None:
    if not graph.input_names:
        return None
    shape = graph.value_shapes.get(graph.input_names[0])
    if not shape:
        return None
    return tuple(1 if dim is None or dim <= 0 else int(dim) for dim in shape)


def _calibrate_relu_scales(
    ann_model: nn.Module, batches: list[torch.Tensor], cfg: ConversionConfig
) -> dict[str, Any]:
    relu_scales: dict[str, float] = {}
    relu_batches: dict[str, int] = {}
    hooks = []

    def hook_for(name):
        def _hook(_module, _inputs, output):
            scale = _activation_scale(output.detach(), cfg.scale_mode)
            if name not in relu_scales:
                relu_scales[name] = scale
            else:
                relu_scales[name] = (
                    (1.0 - cfg.momentum) * relu_scales[name] + cfg.momentum * scale
                )
            relu_batches[name] = relu_batches.get(name, 0) + 1

        return _hook

    for name, module in ann_model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_for(name.replace("ops.", ""))))

    with torch.no_grad():
        for x in batches:
            ann_model(x.to(cfg.device))

    for hook in hooks:
        hook.remove()

    relu_scales = {name: max(float(scale), 1.0e-6) for name, scale in relu_scales.items()}
    return {
        "scale_mode": cfg.scale_mode,
        "relu_scales": relu_scales,
        "relu_batches": relu_batches,
        "num_calibration_batches": len(batches),
    }


def _activation_scale(x: torch.Tensor, mode: str | float) -> float:
    if isinstance(mode, str) and mode.endswith("%"):
        return float(np.percentile(x.detach().cpu().numpy(), float(mode[:-1])))
    if isinstance(mode, str) and mode.lower() == "max":
        return float(x.max().item())
    if isinstance(mode, float) and 0.0 < mode <= 1.0:
        return float(x.max().item() * mode)
    raise NotImplementedError(f"Unsupported scale mode: {mode!r}")


def _compare_onnxruntime(
    onnx_path: Path, ann_model: nn.Module, batches: list[torch.Tensor], cfg: ConversionConfig
) -> dict[str, Any]:
    if not cfg.compare_onnxruntime:
        return {"status": "skipped", "reason": "disabled"}
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"status": "skipped", "reason": f"onnxruntime unavailable: {exc}"}

    if not batches:
        return {"status": "skipped", "reason": "no calibration/sample batch"}

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    x = batches[0].detach().cpu()
    expected_batch = input_meta.shape[0] if input_meta.shape else None
    if isinstance(expected_batch, int) and x.shape[0] != expected_batch:
        merged = torch.cat([batch.detach().cpu() for batch in batches], dim=0)
        if merged.shape[0] < expected_batch:
            return {
                "status": "skipped",
                "reason": "sample batch does not match static ONNX batch size",
            }
        x = merged[:expected_batch]
    with torch.no_grad():
        ann_out = ann_model(x.to(cfg.device)).detach().cpu().numpy()
    ort_out = session.run(None, {input_name: x.numpy()})[0]
    diff = np.asarray(ort_out) - np.asarray(ann_out)
    return {
        "status": "ok",
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "allclose_1e_4": bool(np.allclose(ort_out, ann_out, atol=1.0e-4, rtol=1.0e-4)),
    }


def _evaluate_models(
    ann_model: nn.Module,
    snn_model: nn.Module,
    loader: Iterable,
    cfg: ConversionConfig,
    surrogate_ann_model: nn.Module | None = None,
) -> dict[str, Any]:
    ann_correct = 0
    surrogate_ann_correct = 0
    snn_correct = 0
    total = 0
    acc_curve = torch.zeros(cfg.t, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(cfg.device).float(), batch[1].to(cfg.device)
            ann_out = ann_model(x)
            ann_correct += (ann_out.argmax(dim=1) == y).sum().item()
            if surrogate_ann_model is not None:
                surrogate_ann_out = surrogate_ann_model(x)
                surrogate_ann_correct += (
                    surrogate_ann_out.argmax(dim=1) == y
                ).sum().item()
            functional.reset_net(snn_model)
            snn_sum = None
            for t in range(cfg.t):
                out = snn_model(x)
                snn_sum = out if snn_sum is None else snn_sum + out
                acc_curve[t] += (snn_sum.argmax(dim=1) == y).sum().item()
            snn_correct += (snn_sum.argmax(dim=1) == y).sum().item()
            total += y.numel()
    ann_metric = ann_correct / total if total else 0.0
    snn_metric = snn_correct / total if total else 0.0
    surrogate_ann_metric = (
        surrogate_ann_correct / total if total and surrogate_ann_model is not None else None
    )
    return {
        "ann_metric": ann_metric,
        "snn_surrogate_ann_metric": surrogate_ann_metric,
        "snn_metric": snn_metric,
        "relative_metric": snn_metric / ann_metric if ann_metric else 0.0,
        "target_relative_metric": cfg.target_relative_metric,
        "passes_target": bool(ann_metric and snn_metric >= cfg.target_relative_metric * ann_metric),
        "snn_accuracy_curve": (acc_curve / total).tolist() if total else [],
        "num_samples": total,
    }


def _write_artifacts(
    output_dir: Path,
    ann_model: nn.Module,
    snn_model: nn.Module,
    cfg: ConversionConfig,
    calibration_stats: dict[str, Any],
    report: dict[str, Any],
) -> None:
    torch.save(ann_model, output_dir / "ann_model.pt")
    torch.save(snn_model, output_dir / "snn_model.pt")
    (output_dir / "conversion_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "calibration_stats.json").write_text(
        json.dumps(calibration_stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_script(output_dir / "run_inference.py", _RUN_INFERENCE_SCRIPT)
    _write_script(output_dir / "evaluate.py", _EVALUATE_SCRIPT)


def _write_script(path: Path, content: str) -> None:
    path.write_text(content.lstrip(), encoding="utf-8")


def _op_counts(nodes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


_RUN_INFERENCE_SCRIPT = r'''
import argparse
import torch
from spikingjelly.activation_based import functional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ann", "snn"], default="snn")
    parser.add_argument("--input", required=True, help="Path to a torch tensor .pt file")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    model_path = "ann_model.pt" if args.model == "ann" else "snn_model.pt"
    model = torch.load(model_path, map_location="cpu", weights_only=False).eval()
    x = torch.load(args.input, map_location="cpu", weights_only=False)
    if args.model == "ann":
        out = model(x)
    else:
        functional.reset_net(model)
        out = None
        for _ in range(args.steps):
            y = model(x)
            out = y if out is None else out + y
    print(out)


if __name__ == "__main__":
    main()
'''


_EVALUATE_SCRIPT = r'''
import json
import argparse
import torch
from spikingjelly.activation_based import functional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="Path to a .pt file containing (x, y)")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    ann = torch.load("ann_model.pt", map_location="cpu", weights_only=False).eval()
    snn = torch.load("snn_model.pt", map_location="cpu", weights_only=False).eval()
    x, y = torch.load(args.batch, map_location="cpu", weights_only=False)
    with torch.no_grad():
        ann_out = ann(x)
        functional.reset_net(snn)
        snn_sum = None
        for _ in range(args.steps):
            out = snn(x)
            snn_sum = out if snn_sum is None else snn_sum + out
    result = {
        "ann_accuracy": float((ann_out.argmax(1) == y).float().mean()),
        "snn_accuracy": float((snn_sum.argmax(1) == y).float().mean()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''
