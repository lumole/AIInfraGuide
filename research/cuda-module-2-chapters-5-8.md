# CUDA 模块第二模块第 5–8 章补全研究笔记

> 调研日期：2026-10-07  
> 目标：为《模块二：CUDA 编程与算子优化》的第 5、6、7、8 章补写提供可直接引用的高信任资料和教程主线。  
> 资料选择：优先 NVIDIA、PyTorch、Triton、vLLM、FlashAttention、Apache TVM、OpenXLA 的官方文档/源码，以及对应算法的一手论文。博客或二手解读不作为关键结论依据。

## 1. 现状审计与补写建议

仓库当前对应目录为 `docs/guides/模块二-CUDA编程与算子优化/`。

| 章节 | 当前文件 | 当前状态 | 补写优先级 |
|---|---|---|---|
| 第 5 章 Softmax 与算子融合 | 章节简介、`5.1`、`5.2` | 两篇实现文章较长，但章节页没有导航；融合边界、验证方法和库实现缺少 | 高 |
| 第 6 章 Attention | 章节简介、`6.1`、`6.2` | FlashAttention V1/V2 解释较完整；缺少 PyTorch SDPA 选择、FlashAttention-3、PagedAttention/Decode 的教程页 | 高 |
| 第 7 章 AI 编译器 | 仅章节简介 | 没有可读正文和动手实验 | 最高 |
| 第 8 章 性能分析工具链 | 仅章节简介 | 没有命令、指标解释、从 trace 到 kernel 的完整工作流 | 最高 |

建议每章保留一个总览页，再将每个小节拆成独立 Markdown 页面。每个小节采用“问题 → 最小可运行例子 → 观察指标 → 常见错误 → 进一步阅读”的顺序。这样比把四章继续堆在单个文件里更适合初学者顺读，也符合仓库已有的独立小节 URL 结构。

## 2. 第 5 章：Softmax 与算子融合

### 2.1 建议的教程小节

1. **Safe Softmax：先保证正确性**：从 `exp(x)` 溢出开始，证明减去行最大值不改变结果；用 CPU/PyTorch 参考实现做误差校验。
2. **Block/warp 规约**：一个 block 处理一行，分别实现 max-reduce、sum-reduce；解释 `__shfl_down_sync` 的 mask 和部分 warp 的边界。
3. **Online Softmax**：从 `(m, l)` 状态的合并公式出发，说明一遍扫描只同时得到统计量；如果不缓存输入，输出通常仍需要再次读取输入，不能笼统宣传为“一次读完”。
4. **访存与向量化**：比较标量加载、`float4`/`half2` 加载，明确对齐、尾部元素和寄存器压力；用有效带宽而非单次机器测量宣称收益。
5. **融合的边界**：示范 `scale + mask + softmax` 或 `bias + activation` 融合；解释融合减少 kernel launch 和中间张量读写，但会增加寄存器/共享内存占用。
6. **库实现与验证**：对照 cuDNN softmax、PyTorch `scaled_dot_product_attention`，用随机输入、极端值、不同 dtype 检查 `max_abs_error`、`allclose` 和 NaN。

### 2.2 一手资料与可引用结论

| 资料 | 可支撑的结论 | 链接 |
|---|---|---|
| NVIDIA CUDA C++ Programming Guide | 线程层次、同步、warp shuffle、异步执行和数学函数的语义；实现规约时必须遵守 warp/block 同步规则 | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) |
| NVIDIA CUDA C++ Best Practices | 合并访存、对齐、共享内存 bank conflict、有效带宽计算方式 | [Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations) |
| NVIDIA CUB 官方文档 | `BlockReduce` 等可复用规约原语，适合作为“手写 kernel 与库实现”的对照 | [CCCL/CUB BlockReduce](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html) |
| Milakov & Gimelshein 一手论文 | Online normalizer 的 `(m, l)` 递推，以及与 safe softmax 的等价性；应明确在线统计量和最终输出写回是两个问题 | [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) |
| NVIDIA cuDNN 文档 | 库级 pointwise/reduction/normalization 与 attention API 的输入布局、数据类型和算法接口，适合作为工程实现的基准 | [cuDNN Operations](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/operations.html)、[Pointwise](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Pointwise.html)、[Normalizations](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Normalizations.html) |
| Triton 官方教程 | 用 block-level DSL 实现 fused softmax，并展示 mask、`tl.load`/`tl.store` 和基准测试 | [Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) |

实现 CUDA 版时还可直接引用 CUDA 的[数学函数与浮点语义](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)和[执行模型](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html)；不要只引用总目录，否则读者难以定位 `expf`、warp 同步和 mask 语义。

### 2.3 对现有正文的校准

- `5.2-CUDA Online Softmax实现.md` 的“减少到 2 次读取”表述应保留条件：只维护 `m/l` 时可将 max 与 sum 合并；为了输出每个 softmax 元素，仍需第二遍读取，除非输入行能放入寄存器/片上存储。论文中的算法和实现应分开讲。
- `5.1` 中 A100 带宽、利用率和版本耗时属于特定形状与编译选项的实验结果，应标注 commit、CUDA 版本、dtype、时钟和重复次数，避免把单机数字当作普适结论。
- “融合一定更快”不成立：融合后的寄存器使用、occupancy 和可复用性需要用 profiler 验证。可将“融合前后中间张量字节数”作为第一项可解释指标。

## 3. 第 6 章：Attention 算子

### 3.1 建议的教程小节

1. **标准 Attention 基线**：写出 `QK^T → scale/mask → softmax → PV`，用小矩阵逐元素验证，测量中间 `S/P` 的显存。
2. **FlashAttention V1**：只保留理解所需的 IO-aware、tiling、online softmax 和反向重计算；用伪代码和一个小尺寸 Triton 版本连接数学与代码。
3. **FlashAttention V2**：解释 Q/KV 维度的工作划分、循环顺序和 causal block skip；性能结论引用论文图表，不自行外推。
4. **FlashAttention-3/Hopper**：专门说明这是 Hopper 优化路径，涉及 TMA、WGMMA、异步流水线和 FP8；不要让读者误以为 A100 可直接运行同一 kernel。
5. **PyTorch SDPA 的工程选择**：展示 `torch.nn.functional.scaled_dot_product_attention` 的接口和 backend 选择；说明 PyTorch 会在输入约束满足时选择 flash/memory-efficient/math backend，并用日志或 profiler 观察实际路径。
6. **Decode、PagedAttention 与 FlashInfer**：区分训练/Prefill 与单 token Decode；先解释 KV cache 的逻辑块映射，再进入 vLLM kernel/FlashInfer 的 API，附最小 benchmark 维度表。

### 3.2 一手资料与可引用结论

| 资料 | 可支撑的结论 | 链接 |
|---|---|---|
| FlashAttention V1 一手论文 | IO-aware tiling、在线 softmax、避免物化完整注意力矩阵、反向重计算和 IO 复杂度分析 | [FlashAttention (NeurIPS 2022)](https://arxiv.org/abs/2205.14135) |
| FlashAttention V2 一手论文 | 更好的并行划分、减少非矩阵乘工作、Q 维度分工和 causal 处理 | [FlashAttention-2](https://arxiv.org/abs/2307.08691) |
| FlashAttention 官方源码 | CUDA/Triton 实际 kernel、版本支持、编译与硬件限制；实现细节应以源码和 README 为准 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| FlashAttention-3 官方 Hopper 实现 | Hopper 专用 TMA/WGMMA/异步流水线与 FP8 路径；用于单独的“硬件特化”小节 | [FlashAttention Hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) |
| NVIDIA CUDA Tile Kernels | 适合把 Attention 的 tiling 从抽象公式连接到 CUDA 的 tile 编程模型 | [Writing Tile Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html) |
| NVIDIA CUDA 异步执行/流水线 | 解释 Hopper 路径中异步拷贝、pipeline 和计算重叠的 CUDA 语义 | [Asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)、[Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) |
| PyTorch 官方 SDPA API | 统一 Attention API、参数语义、可用 backend 和 dropout 行为；教学示例以该 API 为基线 | [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| PyTorch SDPA backend 文档 | 如何通过 `sdpa_kernel`/backend 设置和约束排查实际使用的实现 | [SDPA implementations](https://docs.pytorch.org/docs/stable/nn.attention.html) |
| vLLM 官方设计文档 | PagedAttention 的逻辑 block、物理 block、block table 与 KV cache 映射关系 | [vLLM PagedAttention design](https://docs.vllm.ai/en/stable/design/paged_attention/) |
| vLLM 官方源码 | Decode attention kernel 的线程映射、KV block 读取和 dtype/头数分支；适合作为源码阅读练习 | [vLLM attention kernels](https://github.com/vllm-project/vllm/tree/main/csrc/attention) |
| FlashInfer 官方文档 | Serving 场景中 prefill/decode、paged KV cache 与批量变长请求的 API 组织方式 | [FlashInfer documentation](https://docs.flashinfer.ai/) |

### 3.3 对现有正文的校准

- FlashAttention 的 HBM 复杂度不能简单写成“从 `O(N²)` 降到 `O(N)`”。准确说法取决于 SRAM 大小、tile 大小和是否计算前向/反向；论文给出的是 IO 上界/渐近分析，正文应引用论文中的公式和假设。
- FlashAttention-3 的 TMA/WGMMA 代码是 Hopper 路径；A100 的 V1/V2 实验数字不能直接用于 V3。
- PagedAttention 主要解决 KV cache 的分页管理和 Decode 访问，不等同于训练阶段的 FlashAttention；章节中应把“训练 Attention”和“Serving Decode Attention”分成两条线。

## 4. 第 7 章：AI 编译器

### 4.1 建议的教程小节

1. **从 Python 函数到 GPU kernel**：同一个 elementwise/reduction 例子分别用 CUDA、Triton、PyTorch 写，比较抽象层次和可控性。
2. **Triton Block-level 模型**：`program_id`、块指针/索引、mask、`tl.load/store`、`tl.dot`、autotune；复现官方 fused softmax，而不是只讲 API 名称。
3. **torch.compile 全链路**：TorchDynamo 捕获 Python 字节码图，AOTAutograd 处理训练图，TorchInductor 生成后端代码；用 `fullgraph=True` 触发 graph break 教学实验。
4. **Graph Break 排查**：展示 `TORCH_LOGS=graph_breaks`、`torch._dynamo.explain` 和编译缓存；强调“能编译”不等于“更快”。
5. **Inductor 性能观察**：使用 PyTorch profiler/Nsight 查看编译前后 kernel 数、融合、首次编译时间与稳态时间。
6. **TVM 与 XLA 定位**：只做概念和最小例子，明确 TVM 的 TensorIR/跨硬件编译路线与 XLA 的 HLO/整体图优化路线，不把它们和 Triton 当成同一层工具。

### 4.2 一手资料与可引用结论

| 资料 | 可支撑的结论 | 链接 |
|---|---|---|
| Triton 官方教程索引 | 官方教学顺序、向量加法、矩阵乘、融合 softmax、layer norm 等完整例子 | [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) |
| Triton fused softmax 教程 | block-level 索引、mask、片上缓存和与 PyTorch 基线的 benchmark 方法 | [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) |
| PyTorch `torch.compile` 总览 | 编译入口、模式、后端和编译/缓存的用户级语义 | [torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html) |
| TorchDynamo 官方概览 | Python 字节码图捕获、guards、graph break 的成因和排查入口 | [Dynamo core concepts](https://docs.pytorch.org/docs/stable/compile/programming_model.dynamo_core_concepts.html) |
| PyTorch 编译器调试/性能文档 | Inductor 生成代码、profiling、编译耗时和 graph break 的实用检查方法 | [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) |
| Apache TVM 官方文档 | TensorIR、算子调度和跨硬件编译的官方学习入口 | [TVM docs](https://tvm.apache.org/docs/) |
| OpenXLA 官方文档 | XLA/HLO 的定位、编译流程和支持的前端/后端生态 | [OpenXLA: XLA](https://openxla.org/xla) |

### 4.3 教学实验建议

用 `y = a * x + b` 和 row-wise softmax 两个例子即可覆盖大部分概念。每个实验都记录：正确性误差、首次调用时间、预热后平均时间、kernel 数量、显存读写估计和 graph break 日志。不要使用未经固定版本的“提升 2 倍”数字作为结论。

## 5. 第 8 章：性能分析工具链

### 5.1 建议的教程小节

1. **先定义测量问题**：区分端到端延迟、kernel 时间、吞吐、GPU 利用率和显存带宽；先固定输入形状、dtype、warmup 和同步方式。
2. **Nsight Systems 宏观 trace**：用 CLI 抓一轮训练/推理，解释 CPU thread、CUDA API、kernel、memcpy、NVTX 区域和 GPU idle gap；从空洞定位到调用栈。
3. **Nsight Compute kernel 下钻**：对单个 kernel 采集 `launch__*`、SM throughput、DRAM throughput、occupancy、warp stall 和 Speed of Light (SOL) 相关 section；解释一次只看一个问题。
4. **PyTorch Profiler**：用 `schedule(wait/warmup/active)`、`record_function`、`on_trace_ready=tensorboard_trace_handler` 捕获稳定窗口；区分 self CPU、CUDA time 和 kernel launch。
5. **从宏观到微观的闭环**：Systems 找到慢迭代/空洞 → Profiler 找到算子 → Compute 找到内存/计算/同步瓶颈 → 修改 kernel → 重新测量。
6. **常见误区**：异步 CUDA 未同步、首次编译/缓存污染、profiling 开销、采集全部 kernel 导致数据过大、把 occupancy 当作性能目标。

### 5.2 一手资料与可引用结论

| 资料 | 可支撑的结论 | 链接 |
|---|---|---|
| NVIDIA Nsight Systems User Guide | 系统级时间线、CUDA API/kernel/memcpy、NVTX、CLI 抓取和导出流程 | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) |
| NVIDIA Nsight Compute Profiling Guide | section/set、指标、规则、roofline/SOL、occupancy 和 warp stall 的解释 | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) |
| NVIDIA Nsight Compute CLI | `ncu` 的 kernel 过滤、section、报告导出和命令行自动化 | [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) |
| NVIDIA CUDA Best Practices | 有效带宽、合并访存、同步与性能测量的通用方法 | [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) |
| NVIDIA NVTX 文档 | 给应用阶段和算子加标记，使 Systems/Compute 时间线可读 | [NVTX documentation](https://nvidia.github.io/NVTX/) |
| NVIDIA CUPTI 文档 | 需要程序化采集时使用 CUDA Profiling Tools Interface；可作为 Nsight 之后的进阶阅读 | [CUPTI](https://docs.nvidia.com/cupti/) |
| PyTorch Profiler API | 活动类型、schedule、profile key、trace 导出和 TensorBoard 集成 | [torch.profiler](https://docs.pytorch.org/docs/stable/profiler.html) |
| PyTorch 官方 profiler recipe | 初学者可运行的 `schedule`、TensorBoard 和 CUDA 活动示例 | [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) |
| PyTorch Inductor profiling 文档 | `torch.compile` 前后 profile 的方式及编译开销的识别 | [Inductor profiling](https://docs.pytorch.org/docs/stable/torch.compiler_inductor_profiling.html) |

### 5.3 可直接交给读者的最小命令

命令应在教程中说明版本和目标进程，示例只作为起点：

```bash
# 宏观：抓取 Python 脚本的 CUDA/NVTX 时间线
nsys profile --trace=cuda,nvtx,osrt -o nsys_report python train_one_iter.py

# 微观：只采集指定 kernel，减少采集开销
ncu --set full --kernel-name-base function \
    --kernel-name 'regex:softmax.*' -o ncu_report python benchmark_softmax.py
```

PyTorch profiler 示例应加入 CUDA 同步和 warmup，报告中同时展示 `key_averages().table(sort_by="cuda_time_total")` 与 TensorBoard trace。Nsight 命令选项随版本变化，发布前应以目标机 `nsys --help`/`ncu --help` 和对应版本文档复核。

## 6. 建议的章节间依赖关系

```text
第 2 章访存/occupancy
        ↓
第 3 章 Reduce ──→ 第 5 章 Softmax/融合 ──→ 第 6 章 Attention
        ↓                                      ↓
第 8 章 Nsight 工具链  ←────────────── 第 7 章 Triton/torch.compile
```

第 8 章应作为贯穿式实验工具，而不是最后才第一次出现。第 7 章的 Triton fused softmax 可以复用第 5 章的数学基线；第 6 章的 FlashAttention 则用来展示“算法、内存和编译器抽象”如何组合。

## 7. 资料使用说明

- 论文用于证明算法和复杂度；官方源码/文档用于说明当前 API、硬件限制和命令。
- 版本敏感内容（PyTorch backend、FlashAttention-3、Nsight 指标名）写作时应标明文档访问日期或软件版本。
- 不建议把知乎、个人博客、未经复现的 benchmark 作为正文关键依据；它们最多放在“延伸阅读”。

### 7.1 现有页面的链接维护

已有 `5.1`/`5.2` 页面中的 CUDA 文档旧锚点虽然有时仍返回 200，但新版站点会丢失锚点定位。补写时应直接替换为稳定的专题页：

- 数学函数：`https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html`
- 执行模型与 warp：`https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html`
- cuDNN 旧链接 `.../api/cudnn-ops-library.html` 已返回 404；改用 [cuDNN Attention](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)、[Pointwise](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Pointwise.html) 或 [Normalizations](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Normalizations.html)。
