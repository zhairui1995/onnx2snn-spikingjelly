# Current ONNX-to-SNN State

Last reviewed: 2026-06-24

## Stage

Experimental ONNX-to-SNN conversion path is implemented and covered by focused unit tests, demos and smoke checks. It is not full ONNX support.

## Evidence Available

- Public implementation: `spikingjelly.activation_based.onnx2snn`.
- Current handoff records 40 explicitly supported ONNX operators and classification-oriented model coverage.
- Latest handoff verification records:
  - `24 passed` for `test/activation_based/test_onnx2snn.py`;
  - `pip check` with no broken requirements;
  - `git diff --check` passed.
- README records a CIFAR-10 ResNet18 example with ANN metric `0.9375`, SNN metric `0.90625`, relative metric `0.9667`; verify the underlying artifacts before external use.

## Main Risk

Operator-level and synthetic coverage may be mistaken for end-to-end model or task support. Real calibration/evaluation evidence remains the gate for 95% relative-performance claims.

## Current Boundary

- Classification CNN/MLP/VGG/ResNet-style paths are primary.
- ReLU-to-IFNode is the primary ANN-SNN conversion.
- Tanh is ANN reconstruction support only.
- Detection, Transformer, RNN, quantized graph, custom ops and full ONNX coverage remain unsupported claims.

## Next Concrete Action

Before expanding support, rerun the focused suite in the only active workspace:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/spikingjelly/bin/python -m pytest -q test/activation_based/test_onnx2snn.py
```

For a new model, inspect its graph/operators/shapes first, then add a focused test and real-data calibration/evaluation plan before changing the support claim.
