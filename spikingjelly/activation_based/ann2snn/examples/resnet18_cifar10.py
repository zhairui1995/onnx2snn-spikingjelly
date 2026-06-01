import argparse
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm
import spikingjelly.activation_based.ann2snn as ann2snn
from spikingjelly.activation_based.ann2snn.sample_models import cifar10_resnet


def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
            else:
                for m in net.modules():
                    if hasattr(m, "reset"):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
            correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
        acc = correct / total
        print("Validating Accuracy: %.3f" % (acc))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-dir", default="/Users/cvue/Documents/datasets")
    parser.add_argument(
        "--checkpoint",
        default=(
            "/Users/cvue/Documents/github_zr/spikingjelly/onnx_files/"
            "SJ-cifar10-resnet18_model-sample.pth"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--calib-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--mode", default="99.9%")
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args()

    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = args.device
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    T = args.timesteps

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    model = cifar10_resnet.ResNet18()
    checkpoint = Path(args.checkpoint).expanduser()
    if args.download_checkpoint or not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        ann2snn.download_url(
            "https://ndownloader.figshare.com/files/26676110",
            str(checkpoint),
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True, transform=transform, download=True
    )
    if args.calib_size is not None:
        train_data_dataset = torch.utils.data.Subset(
            train_data_dataset, range(min(args.calib_size, len(train_data_dataset)))
        )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False, transform=transform, download=True
    )
    if args.eval_size is not None:
        test_data_dataset = torch.utils.data.Subset(
            test_data_dataset, range(min(args.eval_size, len(test_data_dataset)))
        )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    print("ANN accuracy:")
    ann_acc = val(model, device, test_data_loader)
    print(f"Converting with mode={args.mode}...")
    model_converter = ann2snn.Converter(mode=args.mode, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print("SNN accuracy:")
    snn_acc = val(snn_model, device, test_data_loader, T=T)
    print(
        {
            "ann_accuracy": ann_acc,
            "snn_accuracy": snn_acc,
            "relative_accuracy": snn_acc / ann_acc if ann_acc else 0.0,
            "timesteps": T,
            "mode": args.mode,
            "calib_size": len(train_data_dataset),
            "eval_size": len(test_data_dataset),
        }
    )


if __name__ == "__main__":
    main()
