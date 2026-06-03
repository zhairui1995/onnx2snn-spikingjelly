# ONNX-to-SNN Support Handoff

This document hands off the current state of the experimental
`spikingjelly.activation_based.onnx2snn` module.

## Current Scope

- Workspace: `/Users/cvue/Documents/github_zr/spikingjelly`
- Fork remote: `zhairui1995/onnx2snn-spikingjelly`
- Latest verified commit when this handoff was written: `8ee2c67a`
- Route: `ONNX -> Canonical Graph IR -> executable ANN + structured ANN + SNN`
- Target domain: classification-oriented ANN-to-SNN conversion.
- Current verified datasets/styles: MNIST-style and CIFAR10-style classification.

The current implementation should be described as an experimental conversion
path, not as full ONNX coverage.

## Public Entry Points

The main public imports are exposed from:

`spikingjelly.activation_based.onnx2snn`

Important APIs:

- `convert_onnx_to_snn(...)`
- `ConversionConfig`
- `ConversionArtifacts`
- `UnsupportedONNXError`
- `analyze_patterns(...)`
- `pattern_report(...)`
- `build_structured_ann_model(...)`
- `BasicBlock`
- `BottleneckBlock`
- `StructuredOnnxGraphModule`

## Supported ONNX Operators

Current explicit ONNX operator support: **40 operators**.

| Category | Operators |
|---|---|
| Arithmetic / elementwise | `Abs`, `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Max`, `Min`, `Clip` |
| Comparison / selection | `Equal`, `Greater`, `Less`, `Where` |
| Neural network layers | `Conv`, `Gemm`, `MatMul`, `BatchNormalization`, `Relu`, `Tanh`, `Dropout` |
| Pooling | `AveragePool`, `MaxPool`, `GlobalAveragePool`, `GlobalMaxPool` |
| Tensor shape / transform | `Flatten`, `Reshape`, `Shape`, `Squeeze`, `Unsqueeze`, `Transpose`, `Concat`, `Split`, `Slice`, `Gather`, `Expand`, `Pad` |
| Constants / identity / dtype | `Constant`, `Identity`, `Cast` |
| Reductions | `ReduceMean` |

Full sorted list:

```text
Abs
Add
AveragePool
BatchNormalization
Cast
Clip
Concat
Constant
Conv
Div
Dropout
Equal
Expand
Flatten
Gather
Gemm
GlobalAveragePool
GlobalMaxPool
Greater
Identity
Less
MatMul
Max
MaxPool
Min
Mul
Neg
Pad
ReduceMean
Relu
Reshape
Shape
Slice
Split
Squeeze
Sub
Tanh
Transpose
Unsqueeze
Where
```

Implementation notes:

- `Conv` supports 1D, 2D, and 3D kernels.
- Grouped convolution is supported through ONNX `Conv` `group`; this covers
  depthwise convolution when `groups == in_channels`.
- `BatchNormalization` supports 1D, 2D, and 3D tensors by choosing the PyTorch
  BN module from inferred input rank.
- `Dropout` is treated as an inference-time no-op.
- `MaxPool` can be replaced by `AveragePool` in the generated SNN path when
  `replace_maxpool_with_avgpool_in_snn=True`.
- `Relu` is the primary ANN-to-SNN conversion activation and is replaced by
  `VoltageScaler -> IFNode(v_reset=None) -> VoltageScaler`.
- `Tanh` is supported for ANN graph reconstruction and ONNX-vs-ANN numerical
  alignment. It is not currently converted into an IF neuron path in the same
  way as `Relu`.

## All Verified Model / Conversion Scenarios

The following have been covered by tests, demos, or direct smoke checks during
this development thread. Some are full PyTorch-to-ONNX model tests; some are
focused ONNX operator smoke graphs.

| Name | Dataset style | Main structure | Status |
|---|---|---|---|
| Toy Conv-BN-ReLU CNN | synthetic / small image | `Conv-BN-ReLU-AvgPool-Linear` | unit tested |
| Toy residual block net | synthetic / small image | Basic residual add block | unit tested |
| MNIST MLP | MNIST | `Flatten-Linear-ReLU` stack | unit tested |
| MNIST LeNet5-ReLU | MNIST | LeNet-style CNN with ReLU | unit tested |
| MNIST LeNet5-Tanh | MNIST | LeNet-style CNN with Tanh | unit tested |
| MNIST Conv-BN-GAP | MNIST | Conv-BN-ReLU + global average pooling style head | unit tested |
| MNIST Tiny Residual | MNIST | small residual CNN | unit tested |
| AlexNet-small | CIFAR10-style | compact AlexNet-like CNN | unit tested |
| Depthwise-CNN | CIFAR10-style | depthwise separable convolution | unit tested |
| SqueezeNet-small | CIFAR10-style | Fire modules with branch `Concat` | unit tested |
| VGG-like MaxPool CNN | CIFAR10-style | sequential VGG-style CNN | unit tested |
| CIFAR10 VGG16_BN path | CIFAR10 | pretrained VGG demo path | demo / experiment path |
| CIFAR10 ResNet18 | CIFAR10 | ResNet BasicBlock | unit tested / demo path |
| CIFAR10 ResNet34 | CIFAR10 | deeper ResNet BasicBlock | pattern tested |
| CIFAR10 ResNet50 | CIFAR10 | ResNet Bottleneck | unit tested |
| 1D Temporal ConvNet graph | synthetic temporal | `Conv1d-BN1d-ReLU-MaxPool1d` | unit tested |
| 3D Conv + GlobalMaxPool graph | synthetic 3D | `Conv3d-BN3d-ReLU-GlobalMaxPool` | unit tested |
| Elementwise operator graph | synthetic tensor | arithmetic, compare, `Cast`, `Where`, `Dropout` | unit tested |
| Shape operator graph | synthetic tensor | `Squeeze/Unsqueeze/Shape/Gather/Expand` | unit tested |
| Slice-Split-Concat graph | synthetic tensor | `Slice`, `Split`, `Concat` | unit tested |
| ImageNet-style ResNet18 | ImageNet-style | ResNet18 with stem MaxPool | smoke tested |
| TVM decompiled ResNet18 ONNX | ImageNet-style | TVM/compiled ONNX ResNet18 | smoke tested |

## Recommended Representative 12 Models

These are the 12 representative non-ImageNet models/scenarios recommended for
reports or support tables. They include the two newly added MNIST SNN-friendly
models.

| # | Model | Dataset style | Why keep it |
|---|---|---|---|
| 1 | MNIST MLP | MNIST | simplest fully connected baseline |
| 2 | MNIST LeNet5-ReLU | MNIST | classic CNN, ReLU-to-IF friendly |
| 3 | MNIST LeNet5-Tanh | MNIST | classic activation variant; validates `Tanh` reconstruction |
| 4 | MNIST Conv-BN-GAP | MNIST | validates lightweight Conv-BN with global-pooling-style head |
| 5 | MNIST Tiny Residual | MNIST | validates residual add on MNIST-scale SNN |
| 6 | AlexNet-small | CIFAR10-style | classic CNN family, compact version |
| 7 | Depthwise-CNN | CIFAR10-style | validates grouped/depthwise convolution |
| 8 | SqueezeNet-small | CIFAR10-style | validates branch concat / Fire module style |
| 9 | VGG-like / VGG16_BN | CIFAR10-style / CIFAR10 | sequential Conv-BN-ReLU-Pool family |
| 10 | CIFAR10 ResNet18 | CIFAR10 | ResNet BasicBlock representative |
| 11 | CIFAR10 ResNet50 | CIFAR10 | ResNet Bottleneck representative |
| 12 | 1D Temporal ConvNet | synthetic temporal | validates Conv1d/BN1d/Pool1d path |

Avoid using both ResNet18 and ResNet34 as representative models in a final
paper-style table unless the point is depth scaling. They are both BasicBlock
ResNets and are structurally similar.

Avoid using both VGG7 and VGG16_BN in a small representative table unless the
point is depth scaling. They are both sequential VGG-family networks.

## Validation Commands

Use the local SpikingJelly environment:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/spikingjelly/bin/python -m pytest -q test/activation_based/test_onnx2snn.py
/opt/homebrew/Caskroom/miniforge/base/envs/spikingjelly/bin/python -m pip check
git diff --check
```

Latest full local result before this handoff:

```text
24 passed
pip check: No broken requirements found.
git diff --check: passed
```

## Important Caveats

- Current `onnx2snn` support is not full ONNX support.
- Current target is classification CNN/MLP/VGG/ResNet-like models.
- Do not claim detection, Transformer, RNN, quantized graph, or custom-op
  support until explicitly implemented and tested.
- Tanh is supported for ANN reconstruction, but the primary SNN replacement
  mechanism is still ReLU-to-IFNode.
- Accuracy claims require real calibration/evaluation loaders. Synthetic smoke
  tests only prove graph conversion and numerical alignment.
- PyTorch ONNX export may fold `BatchNormalization` into `Conv` in eval mode.
  This is normal and should not be misinterpreted as missing model support.

## Suggested Next Steps

1. Add `Sigmoid`, `Softmax`, `ArgMax`, `ReduceSum`, and `ReduceMax` if complete
   ONNX classification heads or post-processing graphs are expected.
2. Add a real `NIN-small` CIFAR10-style model to improve representative model
   diversity through 1x1-heavy convolution blocks.
3. Add a dedicated support matrix table in the README if this fork will be
   shown to collaborators or reviewers.
4. Run real MNIST and CIFAR10 calibration/evaluation experiments before making
   performance preservation claims.
5. Keep new operator claims tied to tests. Prefer adding a small stable ONNX
   graph whenever PyTorch export may optimize an operator away.
