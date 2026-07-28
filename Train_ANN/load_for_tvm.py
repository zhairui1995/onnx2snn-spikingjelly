from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import create_model, get_model_spec, list_models


def load_model_for_tvm(
    model_name: str,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load a trained ANN checkpoint and return an eval-mode PyTorch model.

    The training checkpoints created by ``train.py`` store weights under
    ``model_state_dict``. This helper also accepts a raw state_dict for convenience.
    """

    spec = get_model_spec(model_name)
    device = torch.device(device)
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = create_model(spec.name)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_example_input(
    model_name: str,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    spec = get_model_spec(model_name)
    return torch.randn(batch_size, *spec.input_shape, device=torch.device(device))


def export_onnx_for_tvm(
    model: torch.nn.Module,
    model_name: str,
    output_path: str | Path,
    batch_size: int = 1,
    opset: int = 13,
    device: str | torch.device = "cpu",
) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example_input = make_example_input(model_name, batch_size=batch_size, device=device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
        )
    return output_path


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        print_model_table()
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --list-models is used")

    spec = get_model_spec(args.model)
    model = load_model_for_tvm(args.model, args.checkpoint, device=args.device)
    example_input = make_example_input(args.model, args.batch_size, device=args.device)
    with torch.no_grad():
        output = model(example_input)

    summary = {
        "model": spec.name,
        "dataset": spec.dataset,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "input_shape": [args.batch_size, *spec.input_shape],
        "output_shape": list(output.shape),
        "device": args.device,
        "mode": "eval",
    }

    if args.export_onnx is not None:
        onnx_path = export_onnx_for_tvm(
            model,
            spec.name,
            args.export_onnx,
            batch_size=args.batch_size,
            opset=args.opset,
            device=args.device,
        )
        summary["onnx_path"] = str(onnx_path)

    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Load a trained ANN checkpoint for TVM compilation handoff."
    )
    parser.add_argument("--model", default="cifar10_resnet18")
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--export-onnx", default=None)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--list-models", action="store_true")
    return parser


def print_model_table():
    for spec in list_models():
        print(f"{spec.name}\t{spec.dataset}\t{spec.input_shape}\t{spec.description}")


if __name__ == "__main__":
    main()
