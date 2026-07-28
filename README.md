# ONNX2SNN for SpikingJelly

An experimental, classification-focused conversion path from a pretrained ONNX
model to an executable PyTorch ANN and a SpikingJelly SNN.

```text
ONNX -> Canonical Graph IR -> structured PyTorch ANN -> SpikingJelly SNN
```

This repository is a personal fork based on SpikingJelly. The public entry point
is `spikingjelly.activation_based.onnx2snn`.

## Status

The converter supports classification-oriented CNN, MLP, VGG-style, residual,
grouped/depthwise-convolution, and related graphs. It is not full ONNX support.

The current primary conversion rule is:

```text
SNN metric >= 0.95 * ONNX/ANN baseline metric
```

The following records were measured on the `server-lbz` GPU server with the
full MNIST/CIFAR-10 test split, 1024 calibration samples, and batch size 1 for
the fixed-batch decompiled ONNX files. The ONNX files themselves are not
committed to this repository.

| Model | Dataset | T | Scale | ONNX/ANN | SNN metric | Relative | Result |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `mnist_tiny_residual` | MNIST | 64 | `99.9%` | 0.9970 | 0.9543 | 0.9572 | Pass |
| `mnist_lenet_tanh` | MNIST | 64 | n/a | 0.9928 | 0.9929 | 1.0000 | Tanh reconstruction |
| `cifar10_depthwise_cnn` | CIFAR-10 | 128 | `99.9%` | 0.8942 | 0.8733 | 0.9766 | Pass |

`mnist_lenet_tanh` is intentionally reported separately: Tanh is currently
implemented for ANN graph reconstruction and numerical alignment, not as the
same ReLU-to-IF neuron replacement used by the primary SNN path.

## Installation

Python 3.11 or newer and PyTorch 2.6 or newer are required.

```bash
git clone https://github.com/zhairui1995/onnx2snn-spikingjelly.git
cd onnx2snn-spikingjelly
python -m pip install -e ".[onnx2snn]"
```

The ONNX conversion path uses `onnx` for graph loading and checking. Install
`onnxruntime` as well when ONNX-vs-ANN numerical comparison is required.

## Minimal Conversion

The converter can use a real calibration loader, or it can create a small
synthetic calibration set when no loader is supplied. Real-data calibration is
required for an accuracy claim.

```python
from spikingjelly.activation_based.onnx2snn import (
    ConversionConfig,
    convert_onnx_to_snn,
)

artifacts = convert_onnx_to_snn(
    "onnx_files/rebuild/cifar10_depthwise_cnn.onnx",
    "results/cifar10_depthwise_cnn_T128",
    ConversionConfig(
        input_shape=(1, 3, 32, 32),
        t=128,
        scale_mode="99.9%",
        device="cuda:0",
        compare_onnxruntime=True,
    ),
)

print(artifacts.report)
```

For real calibration and evaluation:

```python
artifacts = convert_onnx_to_snn(
    "model.onnx",
    "results/model",
    ConversionConfig(
        input_shape=(1, 3, 32, 32),
        t=128,
        scale_mode="99.9%",
        device="cuda:0",
    ),
    calibration_loader=train_loader,
    eval_loader=test_loader,
)
```

`calibration_loader` may yield tensors or `(input, label)` pairs. The
`eval_loader` must yield `(input, label)` pairs. If an ONNX graph has a static
batch dimension of 1, use `batch_size=1` for ONNXRuntime evaluation and task
metrics.

## Conversion Artifacts

Each conversion output directory contains:

- `ann_model.pt`: executable ANN reconstructed from the canonical graph.
- `structured_ann_model.pt`: readable structured ANN when patterns are found.
- `snn_model.pt`: SNN generated from the same graph and parameter mapping.
- `conversion_config.json`: conversion settings.
- `calibration_stats.json`: activation scales and calibration coverage.
- `report.json`: operator counts, numerical comparisons, and evaluation metrics.
- `run_inference.py`: inference entry point for a saved tensor.
- `evaluate.py`: ANN/SNN evaluation entry point for a saved `(input, label)` pair.

The loader also restores numeric parameters emitted as ONNX `Constant` nodes,
including tensor weights, scalar values, and integer shape lists. This is
needed for binary-decompiled ONNX graphs whose weights are not stored in the
usual ONNX initializer list.

## Supported Operators

The current explicit operator set includes:

```text
Abs Add AveragePool BatchNormalization Cast Clip Concat Constant Conv Div
Dropout Equal Expand Flatten Gather Gemm GlobalAveragePool GlobalMaxPool
Greater Identity Less MatMul Max MaxPool Min Mul Neg Pad ReduceMean Relu
Reshape Shape Slice Split Squeeze Sub Tanh Transpose Unsqueeze Where
```

Grouped convolution, including depthwise convolution, is supported through the
ONNX `group` attribute. ReLU is converted using voltage scaling and
SpikingJelly `IFNode(v_reset=None)`. MaxPool replacement with AvgPool is
configurable and should be validated on the target model.

Unsupported claims: detection, Transformer, RNN, quantized graphs, custom ops,
and full ONNX coverage have not been established by this project.

## Verification

Run the focused conversion suite from the repository root:

```bash
python -m pytest -q test/activation_based/test_onnx2snn.py
python -m pip check
git diff --check
```

The focused suite includes Constant-embedded parameters, fixed-batch vector
outputs, ONNXRuntime numerical comparison, residual graphs, VGG-style pooling,
grouped convolution, shape operators, and structured ResNet reconstruction.

## Project Documents

- [`NOW.md`](NOW.md): current experimental state and evidence boundary.
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md): validated local/server paths.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): conversion and failure-triage workflow.
- [`docs/EXPERIMENT_POLICY.md`](docs/EXPERIMENT_POLICY.md): metric and claim rules.
- [`ONNX2SNN_SUPPORT_HANDOFF.md`](ONNX2SNN_SUPPORT_HANDOFF.md): support matrix and
  technical caveats.

## License

This fork retains the licensing and attribution files of the upstream
SpikingJelly project. See [`LICENSE`](LICENSE) for the applicable terms.
