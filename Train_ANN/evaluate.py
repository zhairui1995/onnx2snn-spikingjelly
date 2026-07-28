from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from datasets import build_dataloaders
from models import create_model, get_model_spec, list_models


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        for spec in list_models():
            print(f"{spec.name}\t{spec.dataset}\t{spec.input_shape}\t{spec.description}")
        return

    spec = get_model_spec(args.model)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --list-models is used")
    checkpoint_path = Path(args.checkpoint).expanduser()
    device = resolve_device(args.device)

    _, _, test_loader = build_dataloaders(
        dataset_name=spec.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        augment=False,
    )

    model = create_model(spec.name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    loss, accuracy = evaluate(model, test_loader, criterion, device)
    result = {
        "model": spec.name,
        "dataset": spec.dataset,
        "checkpoint": str(checkpoint_path),
        "loss": loss,
        "accuracy": accuracy,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained ANN checkpoint.")
    parser.add_argument("--model", required=False, default="cifar10_resnet18")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", required=False, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
    return total_loss / max(total, 1), correct / max(total, 1)


def resolve_device(device: str):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
