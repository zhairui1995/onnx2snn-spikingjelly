from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from spikingjelly.activation_based.onnx2snn import ConversionConfig, convert_onnx_to_snn


HUB_REPO = "chenyaofo/pytorch-cifar-models"
SUPPORTED_VGG = ("vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn")


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Demo: CIFAR-10 pretrained VGG ONNX -> reconstructed ANN + SNN."
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_VGG,
        default="vgg16_bn",
        help="CIFAR-10 VGG variant from chenyaofo/pytorch-cifar-models.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/Users/cvue/Documents/datasets",
        help="CIFAR-10 download/cache directory.",
    )
    parser.add_argument(
        "--onnx-dir",
        default="/Users/cvue/Documents/github_zr/spikingjelly/onnx_files",
        help="Directory for ONNX and conversion artifacts.",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=256,
        help="Number of training samples used for activation calibration.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=64,
        help="Number of test samples used for the small demo evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--keep-snn-maxpool",
        action="store_true",
        help="Keep MaxPool in the SNN instead of replacing it with AvgPool.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    onnx_dir = Path(args.onnx_dir).expanduser()
    model_entry = f"cifar10_{args.model}"
    onnx_path = onnx_dir / f"{model_entry}.onnx"
    pool_suffix = "maxpool_snn" if args.keep_snn_maxpool else "avgpool_snn"
    output_dir = onnx_dir / f"{model_entry}_onnx2snn_{pool_suffix}_demo"

    onnx_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root=str(dataset_root), train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(dataset_root), train=False, transform=transform, download=True
    )

    source_ann = torch.hub.load(
        HUB_REPO, model_entry, pretrained=True, trust_repo=True
    ).to(args.device).eval()

    if not onnx_path.exists():
        dummy = torch.randn(1, 3, 32, 32, device=args.device)
        torch.onnx.export(
            source_ann,
            dummy,
            str(onnx_path),
            opset_version=13,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )

    calib_loader = DataLoader(
        Subset(train_set, range(min(args.calib_size, len(train_set)))),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    eval_loader = DataLoader(
        Subset(test_set, range(min(args.eval_size, len(test_set)))),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    artifacts = convert_onnx_to_snn(
        onnx_path=onnx_path,
        output_dir=output_dir,
        config=ConversionConfig(
            input_shape=(1, 3, 32, 32),
            t=args.timesteps,
            scale_mode="99.9%",
            device=args.device,
            compare_onnxruntime=True,
            replace_maxpool_with_avgpool_in_snn=not args.keep_snn_maxpool,
        ),
        calibration_loader=calib_loader,
        eval_loader=eval_loader,
    )

    source_ann_metric = evaluate_ann(source_ann, eval_loader, args.device)
    summary = {
        "model": model_entry,
        "onnx_path": str(onnx_path),
        "output_dir": str(output_dir),
        "device": args.device,
        "timesteps": args.timesteps,
        "calib_size": min(args.calib_size, len(train_set)),
        "eval_size": min(args.eval_size, len(test_set)),
        "replace_maxpool_with_avgpool_in_snn": not args.keep_snn_maxpool,
        "source_ann_metric": source_ann_metric,
        "reconstructed_ann_metric": artifacts.report.get("evaluation", {}).get(
            "ann_metric"
        ),
        "snn_surrogate_ann_metric": artifacts.report.get("evaluation", {}).get(
            "snn_surrogate_ann_metric"
        ),
        "snn_metric": artifacts.report.get("evaluation", {}).get("snn_metric"),
        "relative_metric": artifacts.report.get("evaluation", {}).get(
            "relative_metric"
        ),
        "snn_graph_transform": artifacts.report.get("snn_graph_transform"),
        "onnx_vs_reconstructed_ann": artifacts.report.get("onnx_vs_ann"),
    }
    (output_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def evaluate_ann(model, data_loader, device: str) -> float:
    model.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            out = model(x.to(device))
            correct += (out.argmax(dim=1).cpu() == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


if __name__ == "__main__":
    main()
