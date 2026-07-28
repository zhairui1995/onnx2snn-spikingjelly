# Train ANN Models for ONNX-to-SNN Experiments

This folder contains training code for 12 representative ANN models grouped by
dataset. The trained `.pt` checkpoints are intended to be exported later for
TVM compilation and binary decompilation experiments.

## Model Set

### MNIST

| Model name | File | Purpose |
|---|---|---|
| `mnist_mlp` | `models/mnist/mnist_mlp.py` | fully connected baseline |
| `mnist_lenet_relu` | `models/mnist/mnist_lenet_relu.py` | classic CNN with ReLU |
| `mnist_lenet_tanh` | `models/mnist/mnist_lenet_tanh.py` | classic CNN with Tanh |
| `mnist_conv_bn_gap` | `models/mnist/mnist_conv_bn_gap.py` | Conv-BN-ReLU with global average pooling |
| `mnist_tiny_residual` | `models/mnist/mnist_tiny_residual.py` | small residual CNN |

### CIFAR-10

| Model name | File | Purpose |
|---|---|---|
| `cifar10_alexnet_small` | `models/cifar10/cifar10_alexnet_small.py` | compact AlexNet-style CNN |
| `cifar10_depthwise_cnn` | `models/cifar10/cifar10_depthwise_cnn.py` | grouped/depthwise convolution |
| `cifar10_squeezenet_small` | `models/cifar10/cifar10_squeezenet_small.py` | Fire modules with branch concat |
| `cifar10_vgg16_bn` | `models/cifar10/cifar10_vgg16_bn.py` | sequential VGG-style Conv-BN-ReLU-Pool |
| `cifar10_resnet18` | `models/cifar10/cifar10_resnet18.py` | ResNet BasicBlock |
| `cifar10_resnet50` | `models/cifar10/cifar10_resnet50.py` | ResNet Bottleneck |
| `cifar10_nin_small` | `models/cifar10/cifar10_nin_small.py` | 1x1-heavy Network-in-Network style CNN |

## Training

Run from this repository root or from inside `Train_ANN`.

```bash
python Train_ANN/train.py \
  --model cifar10_resnet18 \
  --data-root /path/to/datasets \
  --output-dir /path/to/checkpoints \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --augment
```

The script writes:

- `<output-dir>/<dataset>/<model>/<model>_best.pt`
- `<output-dir>/<dataset>/<model>/<model>_last.pt`

Each `.pt` file stores a `state_dict` plus metadata such as model name, dataset,
input shape, epoch, optimizer settings, and accuracy.

## Evaluation

```bash
python Train_ANN/evaluate.py \
  --model cifar10_resnet18 \
  --checkpoint /path/to/checkpoints/cifar10/cifar10_resnet18/cifar10_resnet18_best.pt \
  --data-root /path/to/datasets
```

## TVM Handoff

Use `load_for_tvm.py` to load a trained `.pt` checkpoint and optionally export a
fixed-shape ONNX file for TVM:

```bash
python Train_ANN/load_for_tvm.py \
  --model cifar10_resnet18 \
  --checkpoint /path/to/checkpoints/cifar10/cifar10_resnet18/cifar10_resnet18_best.pt \
  --batch-size 1 \
  --export-onnx /path/to/onnx/cifar10_resnet18.onnx
```

See `README_TVM_HANDOFF.md` for the colleague-facing handoff notes.

## Notes

- Dataset root is always provided by `--data-root`.
- Training is not run locally by this folder; it is prepared for remote servers.
- Checkpoints are PyTorch `.pt` files using `model_state_dict`.
- Use `python Train_ANN/train.py --list-models` to show supported model names.
