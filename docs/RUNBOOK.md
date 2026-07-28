# ONNX-to-SNN Runbook

Run all commands from:

```bash
cd /Users/cvue/Documents/github_zr/spikingjelly
```

## Focused Verification

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/spikingjelly/bin/python -m pytest -q test/activation_based/test_onnx2snn.py
/opt/homebrew/Caskroom/miniforge/base/envs/spikingjelly/bin/python -m pip check
git diff --check
git status --short
```

## Model Workflow

1. Inspect the ONNX graph, opset, operators, shapes and initializers.
2. Confirm calibration/evaluation loaders and baseline metric.
3. Build/validate Canonical Graph IR.
4. Compare ONNXRuntime output with the structured ANN when ONNXRuntime is available.
5. Generate the SNN from the same graph.
6. Evaluate metric over simulation time and compute the final relative metric.
7. Save artifacts, config, report and runnable command.

## Demo Entrypoints

Repository demos:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/spikingjelly/bin/python tools/demo_onnx2snn_cifar10_resnet18.py
/opt/homebrew/Caskroom/miniforge/base/envs/spikingjelly/bin/python tools/demo_onnx2snn_cifar10_vgg.py
```

Inspect each script's arguments and expected local model/data files before running. A successful demo does not by itself establish general model-family support.

## Failure Triage

Report:

- first unsupported operator or pattern;
- inferred and actual tensor shapes;
- ONNX-vs-structured-ANN numerical divergence;
- ANN-vs-SNN metric over time;
- calibration coverage and voltage-scale behavior;
- smallest reproducible graph/test.
