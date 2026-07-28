# ONNX-to-SNN Experiment Policy

## Required Record

Each consequential conversion/evaluation must record:

- date, branch and commit/snapshot;
- ONNX model source, opset and graph/operator summary;
- dataset and calibration/evaluation split;
- Canonical Graph IR and structured ANN artifacts;
- conversion config, neuron/pooling policy and simulation time;
- ONNXRuntime/ANN baseline metric;
- SNN metric by time step and final relative metric;
- logs, output paths, failures and unsupported features.

## Claim Levels

- `implemented`: code path exists.
- `unit tested`: focused automated test passes.
- `smoke tested`: graph converts/executes on synthetic or limited input.
- `demo path`: repository demo exists and runs under stated prerequisites.
- `real-data validated`: calibration/evaluation on a real dataset supports the metric.

Do not collapse these levels into a generic "supported" claim.

## Acceptance

The project target is:

```text
SNN metric >= 0.95 * ONNX/ANN baseline metric
```

Apply it only with comparable real-data metrics and documented simulation/calibration settings. If no real calibration/evaluation set exists, only smoke-test or numerical-alignment claims are allowed.

## Minimum Coverage

Focused tests should retain:

- toy Conv-BN-ReLU;
- ResNet-style residual graphs;
- VGG-style MaxPool graphs;
- ONNXRuntime-vs-structured-ANN numerical comparison when available.

New operator claims require a stable focused test. New model-family claims require representative end-to-end evidence.

## Quality Prohibitions

- No fabricated references, metrics, logs, commits or support status.
- No inference from synthetic execution to real task accuracy.
- No silent fallback to `onnx2torch` that bypasses the structured route.
- No broad exception swallowing or redundant compatibility code that hides unsupported operators, shape errors or numerical divergence.
- No vague paper wording that obscures whether evidence is unit, smoke, demo or real-data validation.
