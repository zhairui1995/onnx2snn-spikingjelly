from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from datasets import build_dataloaders
from models import create_model, get_model_spec, list_models


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        print_model_table()
        return

    set_seed(args.seed)
    spec = get_model_spec(args.model)
    if args.dataset is not None and args.dataset.lower() != spec.dataset:
        raise ValueError(f"{args.model} is registered for {spec.dataset}, not {args.dataset}")

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser() / spec.dataset / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=spec.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        download=args.download,
        augment=args.augment,
        val_split=args.val_split,
        seed=args.seed,
    )
    eval_loader = val_loader if val_loader is not None else test_loader
    eval_split = "val" if val_loader is not None else "test"

    model = create_model(spec.name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_acc = float(checkpoint.get("best_accuracy", 0.0))

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            f"{eval_split}_loss": eval_loss,
            f"{eval_split}_accuracy": eval_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_start, 3),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        is_best = eval_acc >= best_acc
        if is_best:
            best_acc = eval_acc
        save_checkpoint(
            output_dir / f"{spec.name}_last.pt",
            model,
            optimizer,
            scheduler,
            spec,
            args,
            epoch,
            eval_acc,
            best_acc,
            history,
        )
        if is_best:
            save_checkpoint(
                output_dir / f"{spec.name}_best.pt",
                model,
                optimizer,
                scheduler,
                spec,
                args,
                epoch,
                eval_acc,
                best_acc,
                history,
            )

    print(f"best_accuracy={best_acc:.6f}")
    print(f"checkpoint_dir={output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train representative ANN models.")
    parser.add_argument("--model", default="cifar10_resnet18")
    parser.add_argument("--dataset", default=None, help="Optional safety check.")
    parser.add_argument("--data-root", required=False, default="./data")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default="cosine")
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--list-models", action="store_true")
    return parser


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
    return total_loss / max(total, 1), correct / max(total, 1)


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


def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    spec,
    args,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    history,
):
    checkpoint = {
        "model_name": spec.name,
        "dataset": spec.dataset,
        "input_shape": spec.input_shape,
        "num_classes": spec.num_classes,
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_args": vars(args),
        "history": history,
    }
    torch.save(checkpoint, path)


def resolve_device(device: str):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_model_table():
    for spec in list_models():
        print(f"{spec.name}\t{spec.dataset}\t{spec.input_shape}\t{spec.description}")


if __name__ == "__main__":
    main()
