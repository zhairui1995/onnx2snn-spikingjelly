from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import download_url
from spikingjelly.activation_based.ann2snn.sample_models import cifar10_resnet
from spikingjelly.activation_based.onnx2snn import ConversionConfig, convert_onnx_to_snn


WEIGHT_URL = "https://ndownloader.figshare.com/files/26676110"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Demo: CIFAR-10 ResNet18 ANN checkpoint -> ONNX -> reconstructed ANN + SNN."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="/Users/cvue/Documents/datasets",
        help="CIFAR-10 download/cache directory. / CIFAR-10 下载和缓存目录。",
    )
    parser.add_argument(
        "--onnx-dir",
        default="/Users/cvue/Documents/github_zr/spikingjelly/onnx_files",
        help="Directory for checkpoint, ONNX, and conversion artifacts. / 权重、ONNX 与转换产物目录。",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=64,
        help="Number of training samples used for activation calibration. / 用于激活标定的训练样本数。",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=128,
        help="Number of test samples used for the small demo evaluation. / demo 小评估使用的测试样本数。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Dataloader batch size. / 数据加载 batch size。",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=16,
        help="SNN simulation timesteps for the demo. / demo 中 SNN 推理时间步。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for export/conversion/evaluation. / 导出、转换和评估使用的设备。",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    onnx_dir = Path(args.onnx_dir).expanduser()
    output_dir = onnx_dir / "resnet18_cifar10_onnx2snn_demo"
    checkpoint_path = onnx_dir / "SJ-cifar10-resnet18_model-sample.pth"
    onnx_path = onnx_dir / "SJ-cifar10-resnet18_model-sample.onnx"

    onnx_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Keep preprocessing identical to SpikingJelly's original CIFAR-10 ANN2SNN demo.
    # 保持预处理与 SpikingJelly 原始 CIFAR-10 ANN2SNN 示例一致。
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    # Download CIFAR-10 once into the requested local datasets directory.
    # 将 CIFAR-10 下载到用户指定的本地数据集目录，已有文件会自动复用。
    train_set = torchvision.datasets.CIFAR10(
        root=str(dataset_root), train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(dataset_root), train=False, transform=transform, download=True
    )

    if not checkpoint_path.exists():
        # Download the pretrained ANN checkpoint used by the upstream demo.
        # 下载上游示例使用的预训练 ANN 权重。
        download_url(WEIGHT_URL, str(checkpoint_path))

    # Build the known source ANN only for exporting ONNX.
    # 这里“知道 ResNet18”仅用于生成 demo 的 ONNX 文件；后续转换器只接收 ONNX。
    ann_source = cifar10_resnet.ResNet18().to(args.device).eval()
    state_dict = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    ann_source.load_state_dict(state_dict)

    if not onnx_path.exists():
        # Export with dynamic batch so ONNXRuntime and reconstructed ANN can use any demo batch size.
        # 使用动态 batch 导出，便于 ONNXRuntime 和重建 ANN 在不同 demo batch size 下运行。
        dummy = torch.randn(1, 3, 32, 32, device=args.device)
        torch.onnx.export(
            ann_source,
            dummy,
            str(onnx_path),
            opset_version=13,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )

    # From this point on, the converter treats the model as an unknown ONNX graph.
    # 从这里开始，转换器只把模型当作未知 ONNX 图处理，不依赖 ResNet18 类名或 Python 结构。
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
        ),
        calibration_loader=calib_loader,
        eval_loader=eval_loader,
    )

    # Also record the original Python ANN metric for this demo subset.
    # 额外记录原始 Python ANN 在 demo 子集上的指标，方便对照。
    source_ann_metric = evaluate_ann(ann_source, eval_loader, args.device)
    snn_metric = artifacts.report.get("evaluation", {}).get("snn_metric")
    relative = artifacts.report.get("evaluation", {}).get("relative_metric")
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "output_dir": str(output_dir),
        "source_ann_metric": source_ann_metric,
        "reconstructed_ann_metric": artifacts.report.get("evaluation", {}).get(
            "ann_metric"
        ),
        "snn_metric": snn_metric,
        "relative_metric": relative,
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


def evaluate_snn_once(model, x: torch.Tensor, timesteps: int) -> torch.Tensor:
    # Helper kept for notebook/debug usage. / 供 notebook 或调试时复用的小工具。
    functional.reset_net(model)
    out_sum = None
    for _ in range(timesteps):
        out = model(x)
        out_sum = out if out_sum is None else out_sum + out
    return out_sum


if __name__ == "__main__":
    main()
