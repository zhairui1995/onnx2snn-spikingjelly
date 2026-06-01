# Project Rules for ONNX-to-SNN SpikingJelly Fork

## Project Identity

- This repository is the only active local workspace for the ONNX-to-SNN fork:
  `/Users/cvue/Documents/github_zr/spikingjelly`.
- GitHub fork: `zhairui1995/onnx2snn-spikingjelly`.
- Upstream source: `fangwei123456/spikingjelly`.
- Do not use `/Users/cvue/Documents/spikingjelly` for project work; it is a temporary clone slated for removal.

## Research Goal

- Convert a pretrained ANN stored as an ONNX file into a SpikingJelly SNN.
- Preserve at least 95% relative task performance:
  `SNN metric >= 0.95 * ONNX/ANN baseline metric`.
- First supported target: classification CNN/ResNet/VGG-style models.
- Do not claim support for detection, Transformer, RNN, quantized, or custom-op ONNX models until validated.

## Technical Route

- Main route: `ONNX -> Canonical Graph IR -> structured PyTorch ANN + SpikingJelly SNN`.
- Do not use a flat `onnx2torch` model as the primary model structure for SNN conversion.
- `onnx2torch` may be used only as an oracle/debug fallback.
- The structured ANN is a first-class artifact for inspection, modification, activation calibration, and ONNX-vs-PyTorch numerical alignment.
- The SNN is generated from the same canonical graph and should use SpikingJelly primitives such as `IFNode(v_reset=None)` and voltage scaling.

## Working Habits

- Before changing conversion logic, inspect the ONNX graph, supported operators, tensor shapes, and calibration/evaluation data availability.
- For failures, report root-cause hypotheses, unsupported operators, shape issues, numerical divergence, and the next diagnostic experiment.
- Every conversion run should produce a reproducible output package with model artifacts, config, report, and runnable scripts.
- Keep privacy boundaries: do not write keys, tokens, passwords, cookies, account credentials, private raw user text, or sensitive complete path inventories into docs or memory.

## Validation Standard

- Unit tests must cover toy Conv-BN-ReLU, ResNet-style residual graphs, and VGG-style MaxPool graphs.
- Numerical checks should compare ONNXRuntime baseline with the structured ANN when ONNXRuntime is available.
- SNN checks should report accuracy over simulation time and final relative metric.
- If no real calibration/evaluation set is provided, only smoke-test claims are allowed.
