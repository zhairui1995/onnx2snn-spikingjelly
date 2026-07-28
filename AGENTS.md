# ONNX-to-SNN SpikingJelly Fork Rules

本文件是该 fork 的稳定规则入口。当前用户指令、系统/开发者指令优先于本文件。

## Source Of Truth

- 稳定边界与禁令：`AGENTS.md`
- 当前阶段、证据和下一步：`NOW.md`
- 唯一路径与环境：`docs/ENVIRONMENT.md`
- 测试、demo 和交付命令：`docs/RUNBOOK.md`
- 实验与 claim 规则：`docs/EXPERIMENT_POLICY.md`
- 当前能力明细：`ONNX2SNN_SUPPORT_HANDOFF.md`

重要任务前读取 `NOW.md`、最小必要实现/测试和真实 diff。涉及支持范围或性能时，同时核对 handoff、README、测试和真实结果。

## Workspace Identity

- 唯一有效本地工作区：`/Users/cvue/Documents/github_zr/spikingjelly`
- Fork: `zhairui1995/onnx2snn-spikingjelly`
- Upstream: `fangwei123456/spikingjelly`
- `/Users/cvue/Documents/spikingjelly` 是禁用旧目录，不得用于读取、修改、测试、提交或生成结果。

## Research Boundary

- Goal: convert a pretrained ANN stored as ONNX into a SpikingJelly SNN.
- Target: `SNN metric >= 0.95 * ONNX/ANN baseline metric`，仅在真实 calibration/evaluation 数据支持时判断。
- Current primary domain: classification CNN/MLP/VGG/ResNet-style models.
- 未验证前不得声称 detection、Transformer、RNN、quantized graph、custom-op 或 full ONNX support。

## Working Model

- 不设固定 Agent 分工、互审链或指定思维路线。当前执行者可端到端完成图分析、实现、测试、实验和文档。
- 可以挑战当前 IR、operator support、calibration、neuron replacement 和 95% target 的实现假设，但必须提出证据或可证伪测试。
- 历史 handoff 和 README 是当前证据快照，不是不可质疑的路线限制。

## ANN-SNN Technical Boundary

主路线：

`ONNX -> Canonical Graph IR -> structured PyTorch ANN + SpikingJelly SNN`

强制边界：

- 不得以 flat `onnx2torch` model 作为主要 SNN conversion structure；它只能作为 oracle/debug fallback。
- Structured ANN 是一等产物，用于图检查、修改、activation calibration 和 ONNX-vs-PyTorch numerical alignment。
- SNN 必须从同一 canonical graph 生成，保持拓扑和参数映射可追溯。
- Primary ReLU conversion uses voltage scaling and SpikingJelly `IFNode(v_reset=None)`.
- `Tanh` 当前只支持 ANN graph reconstruction/numerical alignment，不得声称已按 ReLU 相同方式转换为 IF neuron。
- MaxPool replacement策略必须由真实性能验证决定；不得把可选 replacement 写成普遍正确。
- 新 operator/support claim 必须有稳定测试，必要时使用小型 ONNX graph 防止 export folding 掩盖覆盖缺口。

## Evidence And Claims

- 不得伪造引用、结果、日志、路径、commit、指标或支持状态。
- 引用必须可核验；无法核验时标记 `TODO`/`unverified`，AI 输出不是学术来源。
- Synthetic smoke tests 只能证明图转换、执行或数值对齐，不能支持真实任务性能 claim。
- 95% relative-performance claim 必须同时记录 ONNX/ANN baseline、SNN metric、simulation time、数据集和 calibration/evaluation 设置。
- 明确区分 implemented、unit tested、smoke tested、demo path 和 real-data validated。
- 禁止过度防御性编程：不使用宽泛异常吞噬、静默 fallback、重复检查或无证据兼容层掩盖 unsupported op/shape/numerical failure。
- 禁止过度防御性论文描述：支持边界应直接准确，不以模糊限定词替代 support matrix。

## Engineering And Safety

- 修改 conversion logic 前检查 ONNX graph、operators、tensor shapes、initializers 和 calibration/evaluation 数据。
- 失败报告应包含 root-cause hypotheses、unsupported operators、shape/numerical divergence 和下一诊断测试。
- 每次 conversion 应产生可复现的 model artifacts、config、report 和 runnable script。
- 保留用户及其他人的无关改动；不要回滚、覆盖或清理未授权文件。
- 不删除未跟踪产物或改写 Git 历史，除非用户明确要求。
- 不写入 key、token、密码、cookie、账号凭据、私人原话或敏感路径清单。
- 未经确认，不向第三方上传代码、模型、结果或数据。

## Completion

用中文报告修改文件、验证命令与结果、未验证风险和下一精确命令。
