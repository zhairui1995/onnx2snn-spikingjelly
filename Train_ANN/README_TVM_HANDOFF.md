# TVM Handoff Notes

This folder contains the model definitions and loader needed to open the 12
trained ANN checkpoints before TVM compilation.

## Expected Files

The training result archive should contain:

```text
results/
  logs/
    <model>.log
  mnist/
    <model>/<model>_best.pt
    <model>/<model>_last.pt
  cifar10/
    <model>/<model>_best.pt
    <model>/<model>_last.pt
```

Each `.pt` checkpoint contains:

- `model_name`
- `dataset`
- `input_shape`
- `num_classes`
- `model_state_dict`
- training metadata and history

## Load A Checkpoint

```bash
python Train_ANN/load_for_tvm.py \
  --model cifar10_resnet18 \
  --checkpoint /home/lbz/mac_agent/Train_ANN/results/cifar10/cifar10_resnet18/cifar10_resnet18_best.pt
```

This prints a JSON summary with input and output shapes. The model is loaded in
`eval()` mode on CPU by default.

## Export ONNX For TVM

```bash
mkdir -p /home/lbz/mac_agent/Train_ANN/results/onnx

python Train_ANN/load_for_tvm.py \
  --model cifar10_resnet18 \
  --checkpoint /home/lbz/mac_agent/Train_ANN/results/cifar10/cifar10_resnet18/cifar10_resnet18_best.pt \
  --batch-size 1 \
  --export-onnx /home/lbz/mac_agent/Train_ANN/results/onnx/cifar10_resnet18.onnx
```

Recommended fixed compiler input shapes:

| Dataset | Shape |
|---|---|
| MNIST | `[1, 1, 28, 28]` |
| CIFAR-10 | `[1, 3, 32, 32]` |

## Python API

```python
from Train_ANN.load_for_tvm import load_model_for_tvm, make_example_input

model = load_model_for_tvm(
    "cifar10_resnet18",
    "/home/lbz/mac_agent/Train_ANN/results/cifar10/cifar10_resnet18/cifar10_resnet18_best.pt",
)
x = make_example_input("cifar10_resnet18", batch_size=1)
y = model(x)
```

## Dependencies

Only the training package dependencies are needed to load/export models:

```text
torch
torchvision
numpy
```

TVM installation and target-specific cross compilation can be handled in the
colleague's own TVM environment after ONNX export.
