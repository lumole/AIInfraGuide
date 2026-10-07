# CUDA 模块第 5-8 章外部教程审计

> 审计日期：2026-10-07
> 对照样板：`docs/guides/模块四-推理优化/第9章-性能分析与Benchmark/`
> 审计对象：`docs/guides/模块二-CUDA编程与算子优化/` 的 5.1-8.4 页面
> 目标：优先给读者一篇能从头顺读的外部教程；论文、API 和 User Guide 只承担查证作用。本文不改正文。

## 1. 总体结论

模块四第 9 章的有效写法不是“正文末尾列很多参考资料”，而是每个页面开头直接告诉读者：**先读哪一篇主教程，它解决什么问题**。正文只负责统一术语、补足中文上下文、指出版本边界，并提供本站实验。

模块二现有第 5-8 章已经拆成独立网页，篇幅也不短；主要问题是 15 个小节仍以本站自写正文为主体，外部资料集中堆在末尾，读者无法区分“适合第一次学习的教程”和“只适合查参数的手册”。此外，绝大多数新页面只有 Mermaid 图，没有来自论文、官方博客或 profiler 的真实图示。

建议所有小节采用下面的固定顺序：

1. `## 主教程`：一篇为主，最多两篇；用两三句话说明阅读范围和前置知识。
2. `## 本站导读/实验`：只写外部教程没有覆盖的中文串联、版本差异、正确性检查和实测。
3. `## 补充教程`：仍然要求可顺读、有例子，不放 API 索引页。
4. `## 一手资料`：论文、源码、API、User Guide，用于核对结论。

资料评级：

- **A**：官方或作者团队教程，逻辑连续，有图或可运行代码，可直接作为主教程。
- **B**：高质量教程，但只覆盖小节的一部分，适合作为补充。
- **C**：论文、源码、API、User Guide；权威但不是入门教程，只用于查证。

## 2. 第 5 章：Softmax 与算子融合

### 5.1 CUDA Softmax 朴素实现优化

**建议主教程（A）**

- [Triton: Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：从 PyTorch 基线、行级 program、mask、片上存储一直讲到 benchmark。虽然代码是 Triton，不是 CUDA C++，但它是目前最完整、最容易顺读的官方 Softmax kernel 教程，适合先建立问题模型。

**补充教程（A/B）**

- [NVIDIA: Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)：经典的 CUDA reduction 教程，逐步解释交错寻址、分支、bank conflict、首次加载时规约、warp 展开。5.1 的 max/sum 两次规约应按它的顺序学习。
- [CUDA Best Practices: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)：向量化加载、合并访问和 shared memory 的查证资料，属于 C，不应替代前两篇教程。

**本站需要补的内容**

- 用 CUDA C++ 把 Triton 教程的“一行一个 program”翻译为“一行一个 block”，解释 active mask 和非 32 倍数列宽。
- 不应再虚构或泛化 A100 实验数字；性能表必须来自可复现实验并标记硬件、版本、dtype 和 shape。

**图片建议**

- 引用 NVIDIA reduction PDF 中逐版本规约图，并标注图源；本站另做一张“一行 Softmax 映射到 block/warp”的简图。

### 5.2 CUDA Online Softmax 实现

**建议主教程（B）**

- [Lei Mao: Online Safe Softmax](https://leimao.github.io/blog/Online-Safe-Softmax/)：先回顾 safe softmax，再逐式推导在线最大值和归一化项，篇幅短、逻辑连续，适合初学者。它是高质量二手教程，不是一手论文。

**一手资料（C）**

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)：Online Softmax 原论文，用于核对 `(m, d)` 递推和内存访问次数。
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)：用于理解 Online Softmax 状态如何扩展到 tiled attention，不宜直接作为本节入门主教程。

**本站需要补的内容**

- 现有页面应该保留 CUDA block/warp 合并规约，这是外部教程没有给出的部分。
- 必须区分“max 和 sum 的统计阶段一次扫描”与“输出 softmax 时是否再次读输入”。只有整行保存在寄存器/片上存储中时，才可能消除再次读取。

**图片建议**

- 自制一张状态合并图：两个分片各自产生 `(m, d)`，再用同一公式合并。公式来自论文，图由本站重绘。

### 5.3 CUDA 算子融合实战

**建议主教程（A）**

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：从带宽、kernel launch、operator fusion 和 recomputation 解释“为什么融合”，图多且面向第一次接触性能优化的读者。
- [Triton: Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：接着看一个真实的融合实现和 benchmark。前一篇讲判断方法，这一篇讲落地代码，两者不重复。

**补充资料（C）**

- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)：核对 pointwise 融合和 `torch.compile` 的工程入口。
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：核对有效带宽、同步和计时原则。

**本站需要补的内容**

- 保留当前 RTX 3060 的 CUDA 实测，因为外部教程不会覆盖本站这个 `scale + mask + softmax` 实验。
- 增加 Nsight Systems 的真实 kernel 时间线截图和 Nsight Compute 的寄存器/带宽截图；只有 Mermaid 流程图不足以证明融合确实发生。

## 3. 第 6 章：Attention 算子

### 6.1 FlashAttention V1

**建议主教程（A）**

- [Triton: Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：官方可运行教程，包含 forward、backward、causal mask、autotune 和 benchmark。建议读者先只读前向代码，再回本站理解数学状态。

**补充教程（A/B）**

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：作为前置阅读，先理解融合、重计算和内存带宽为何重要。
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)：一手资料，用于核对 IO 复杂度和反向重计算；不是第一次阅读的主教程。
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)：核对当前安装、硬件支持和实现版本。

**本站需要补的内容**

- 用一个 `4 x 4` 的数值例子串起 tiling、局部 max、缩放旧输出和合并，外部 Triton 教程在这一步跳得较快。
- 删掉“Attention 本质上 memory-bound”这类无条件表述；不同 shape、训练/推理阶段和硬件下可能不同。

**图片建议**

- 使用论文 Figure 1/2 的 IO 路径图并清楚标明论文图号与来源；再配一张本站小矩阵手算图。

### 6.2 FlashAttention V2

**建议主教程（A）**

- [Princeton NLP / Tri Dao: FlashAttention-2](https://princeton-nlp.github.io/flash-atttention-2/)：作者团队写的发布文章，用直观图示解释减少 non-matmul FLOPs、sequence 维并行和 warp 工作划分，明显比直接读论文更适合初学者。

**补充资料（C）**

- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)：核对算法、硬件、shape 和性能数字。
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)：核对当前实现；不要将仓库 README 当作完整教程。

**本站需要补的内容**

- 现有正文应围绕作者文章的三项变化做中文导读，不必重新写一套平行叙事。
- 所有 TFLOPS 和相对 V1 的提升都应明确来自论文的哪张表、哪块 GPU，或替换为本站实测。

**图片建议**

- 直接引用作者文章里的 V1/V2 work partitioning 图，能替代大段文字描述。

### 6.3 FlashAttention-3 与 Hopper

**建议主教程（A）**

- [Together AI / FlashAttention 作者团队：FlashAttention-3](https://www.together.ai/blog/flashattention-3)：作者团队文章从 Hopper 的 WGMMA/TMA、warp specialization、异步流水线讲到 FP8，图示丰富，适合作为本节主线。

**补充资料（C）**

- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08691)：核对算法与性能数据。
- [FlashAttention Hopper 实现](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)：核对 CUDA 版本、H100/H800 限制、构建和 benchmark。
- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) 与 [Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)：只用于查 TMA/pipeline 语义，不作为主教程。

**本站需要补的内容**

- 当前实验机 RTX 3060 不能运行 FA3 的 Hopper 路径。本站只能给出代码阅读和运行命令，不能伪造本机 H100 数据；若没有 H100，性能数字只引用论文并明确来源。

**图片建议**

- Together AI 文章已有 producer/consumer warp、ping-pong schedule 和 FP8 accuracy 图，应优先引用，而不是重新画内容更少的 Mermaid 图。

### 6.4 Flash-Decoding 与 PagedAttention

这个页面实际包含两个独立主题，一篇主教程不能充分覆盖。最合理的做法是拆成 6.4 Flash-Decoding 和 6.5 PagedAttention，原 6.5 顺延；若暂时不拆，至少给两篇并列主教程。

**建议主教程（A）**

- [PyTorch: Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)：从 `query length = 1` 导致并行度不足讲到 KV 维切分和局部 log-sum-exp 合并，是 Flash-Decoding 的作者/实现团队教程。
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)：从 KV cache 浪费和碎片问题引出分页、共享和调度，图示对初学者很友好。

**补充资料（C）**

- [vLLM PagedAttention design](https://docs.vllm.ai/en/stable/design/paged_attention/)：kernel 级实现说明；适合读完博客后查线程/块映射。
- [vLLM attention kernels](https://github.com/vllm-project/vllm/tree/main/csrc/attention)：源码阅读入口。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：核对内存利用率、调度和实验条件。

**本站需要补的内容**

- 把“沿 KV 长度增加计算并行度”和“KV cache 的逻辑页到物理页映射”分成两条叙事，不要混成一种优化。
- RTX 3060 可验证简化的局部 `(m, l, o)` 合并和 block table 寻址，但不能据此给出 vLLM 生产吞吐结论。

**图片建议**

- Flash-Decoding 使用 PyTorch 文章中的序列维切分图；PagedAttention 使用 vLLM 博客的 contiguous/paged KV cache 对比图，并保留许可和图源。

### 6.5 PyTorch SDPA 后端选择

**建议主教程（A）**

- [PyTorch: Accelerating PyTorch Transformers by replacing nn.Transformer with Nested Tensors and torch.compile](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)：从基本 Attention 构造逐步过渡到 SDPA，并解释 padding、causal 和嵌套张量，适合建立工程上下文。
- [PyTorch: Scaled Dot Product Attention tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)：直接演示 SDPA、backend、benchmark 和 `torch.compile`，作为本节操作主教程。

**补充资料（C）**

- [`scaled_dot_product_attention` API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：核对参数、dropout、GQA 和 warning。
- [`torch.nn.attention` backend controls](https://docs.pytorch.org/docs/stable/nn.attention.html)：核对 `sdpa_kernel` 和 backend 枚举。

**本站需要补的内容**

- 保留“如何确认实际走了哪个 backend”和不同 shape 的回退实验，这是教程读者最容易踩坑的地方。
- 后端名字和选择条件随 PyTorch 版本变化，正文应固定版本，不要写成永久规则。

## 4. 第 7 章：AI 编译器

### 7.1 Triton 编程模型与 Fused Softmax

**建议主教程（A）**

- [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)：第一次接触 Triton 时先读，完整解释 `program_id`、offset、mask、load/store 和 launch grid。
- [Triton: Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：紧接着把同一编程模型用于真实 reduction/fusion，并包含 benchmark。

**补充教程（B）**

- [Triton: Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：补 `tl.dot`、block pointer、L2-friendly ordering 和 autotune；不要在 Softmax 主线中一次塞完。

**本站需要补的内容**

- 当前页面可以缩成两篇官方教程的中文阅读地图，再补“CUDA block 与 Triton program 的对应关系”和中文错误排查。
- 当前页面的 Mermaid 语法已在浏览器出现错误，修复后仍应加入官方教程的性能图或本站 benchmark 图。

### 7.2 `torch.compile` 与 Inductor

**建议主教程（A）**

- [PyTorch: Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)：从 eager baseline、编译、性能测量到 graph break，正好覆盖本节主线。

**补充教程（A/B）**

- [PyTorch: A full example of applying torch.compile to a real model](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)：把基础示例扩展到完整模型，并展示实际编译工作流。
- [PyTorch: User-defined Triton kernels with torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)：衔接 7.1 与 7.2，展示 Inductor 与自定义 Triton kernel 的边界。

**查证资料（C）**

- [TorchDynamo Core Concepts](https://docs.pytorch.org/docs/stable/compile/programming_model.dynamo_core_concepts.html)
- [torch.compile Troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- [TorchInductor Profiling](https://docs.pytorch.org/docs/stable/torch.compiler_inductor_profiling.html)

**本站需要补的内容**

- 用一个固定模型展示首次编译时间、稳态时间、kernel 数和 graph breaks；不要仅凭 Python 代码猜测是否融合。

### 7.3 MLIR、TVM 与 XLA 选型

这一节当前范围过大。MLIR 是编译器基础设施，TVM 是端到端机器学习编译栈，XLA 是面向线性代数图的编译器；不存在一篇可靠的初学者教程能同时教会三者并完成选型。建议把 7.3 改成导航页，并拆成三个独立页面。

**MLIR 主教程（A）**

- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)：官方从 AST、Dialect、转换、Lowering 到 LLVM 的完整教程，是理解多层 IR 最合适的起点。

**TVM 主教程（A）**

- [TVM: TensorIR Creation](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)：从张量程序到 TensorIR。
- [TVM: TensorIR Transformation](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html)：接着学习 split、reorder、blockize 等 schedule 变换。

**XLA 主阅读（B/C）**

- [OpenXLA: XLA overview](https://openxla.org/xla)：官方入口，解释 HLO、前后端和整体定位。OpenXLA 目前缺少与 MLIR Toy 同等完整、面向初学者的端到端教程，因此本站需要承担更多导读。
- [StableHLO Specification](https://openxla.org/stablehlo/spec)：只作算子语义查证，不要列为主教程。

**失效链接**

- 当前页面引用的 `https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tutorial.html` 已返回 404，应改为上面的 `tir_creation.html` 和 `tir_transformation.html`。

**图片建议**

- 用三个官方项目的 IR 示例做并排对照，比抽象层次 Mermaid 图更有信息量；图片或代码必须标明来自哪个教程版本。

## 5. 第 8 章：性能分析工具链

### 8.1 Nsight Systems 系统级分析

**建议主教程（A）**

- [NVIDIA: Three Things You Need to Know About Nsight Systems](https://developer.nvidia.com/blog/three-things-you-need-to-know-about-nsight-systems/)：先建立“系统级时间线、低开销采集、从 CPU 到 GPU”的工具边界。
- [NVIDIA: Understanding the Visualization of Overhead and Latency in Nsight Systems](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)：用真实 timeline 解释 launch latency、GPU idle 和 profiler overhead，适合接着实操。

**补充教程（A/B）**

- [NVIDIA: Optimizing CUDA Memory Transfers with Nsight Systems](https://developer.nvidia.com/blog/optimizing-cuda-memory-transfers-with-nsight-systems/)：围绕 memcpy、pageable/pinned memory 和重叠给出完整案例。

**查证资料（C）**

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CLI 选项和版本差异，只作查表。

**本站需要补的内容**

- 使用本站 Softmax 实验生成一张真实 `.nsys-rep` 时间线截图，标注分离/融合 kernel 和空洞；这比文字复述 User Guide 更有价值。

### 8.2 Nsight Compute Kernel 级分析

**建议主教程（A）**

- [NVIDIA: Using Nsight Compute to Inspect your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)：从一个真实 kernel 进入 GUI，解释 source、memory chart、occupancy 和 baseline，对初学者比 Profiling Guide 更合适。
- [NVIDIA: Accelerating HPC Applications with Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)：读完基本界面后，用案例学习 arithmetic intensity 与 memory/compute bound 的判断。

**查证资料（C）**

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：指标、replay 和 section 的权威定义。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：自动化采集和 kernel filter 查表。

**本站需要补的内容**

- 明确 NCU replay 会显著扰动 wall-clock 时间；报告内 metric 用于解释硬件行为，性能 benchmark 仍应使用 CUDA Event/稳定计时。
- 用本站 Softmax kernel 截出 Summary、Memory Workload、Occupancy 三张图，并围绕一个具体问题解读，避免堆指标名。

### 8.3 PyTorch Profiler 与 TensorBoard

**建议主教程（A）**

- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：可直接运行，依次讲 CPU/CUDA activity、shape、memory、stack、trace 导出和 overhead，是本节最合适的主教程。

**补充教程（A/B）**

- [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)：先学会同步和可靠计时，避免把 profiler 时间当 benchmark。
- [PyTorch Profiler API](https://docs.pytorch.org/docs/stable/profiler.html)：查 `schedule`、`on_trace_ready`、activity 等参数，只作查证。

**本站需要补的内容**

- 当前最新官方教程已经不再维护旧的 `tensorboard_profiler_tutorial.html` 页面，不要加入这个已返回 404 的链接。
- 应用同一个 workload 同时输出 `key_averages()` 表和 Chrome trace，并说明 `self_cuda_time_total` 与端到端时间为何不同。

### 8.4 从服务指标到 Kernel 的诊断闭环

**外部资料结论**

没有找到一篇同时覆盖“服务 SLO -> PyTorch operator -> Nsight Systems -> Nsight Compute”的官方初学者教程。这个主题确实需要本站原创串联，但每一层应复用已有主教程，不应重写工具手册。

**建议主学习路径（A）**

1. [NVIDIA: LLM Inference Benchmarking - Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)：先固定 TTFT、ITL、TPOT、吞吐、并发和请求率。
2. [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：定位到 operator/kernel。
3. [NVIDIA: Understanding Overhead and Latency in Nsight Systems](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)：识别 CPU/GPU 空洞和 launch 问题。
4. [NVIDIA: Using Nsight Compute to Inspect your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)：对单个热点 kernel 下钻。

**本站需要补的内容**

- 本节的独特价值应是一份贯穿四层的同一案例，而不是四套工具简介。最好沿用第 5 章的分离/融合 Softmax，保存请求指标、Profiler 表、Systems 时间线和 Compute 报告。
- 与模块四 9.3 内容重复较多。CUDA 模块保留“单机算子到 kernel”，推理模块保留“服务请求到调度/通信/算子”，并互相链接。

## 6. 推荐的落地优先级

### 第一优先级：先建立外部教程入口

给 15 个小节都加 `## 主教程`，但不要机械地放 API 页面。最先处理 7.1、7.2、8.1、8.2、8.3：这些主题已经有质量很高的官方教程，本站大段重写价值较低。

### 第二优先级：用官方图和真实 profiler 图替换空泛图

- 6.1-6.4 优先使用作者/论文的算法和 work partitioning 图。
- 8.1-8.2 必须加入本站实际采集的 Systems/Compute 截图。
- 5.3、7.1 加入 shape-performance 曲线，而不是只给单点表格。

引用外部图片时应检查许可，保留图源、原文链接、作者/项目和许可说明；不能确认转载许可时，按原图信息自行重绘并注明“据某资料重绘”。

### 第三优先级：只原创外部教程没有覆盖的部分

- 5.2：CUDA 中 `(m, d)` 的并行合并和寄存器缓存边界。
- 5.3：RTX 3060 上可复现的融合实验。
- 6.1：小矩阵手算和中文公式串联。
- 6.4：Flash-Decoding 与 PagedAttention 的边界。
- 7.3：三套编译器的层次与选型；建议拆页。
- 8.4：同一案例贯穿服务指标、operator、timeline 和 kernel metric。

## 7. 链接核验说明

上述主教程链接在 2026-10-07 做过可访问性检查；当前网络无法连接 `openxla.org`，因此 OpenXLA 两个链接只按项目官方稳定路径核对，未宣称完成在线访问验证。审计中已排除或替换三个失效入口：

- Fireworks 旧的 Flash-Decoding 博客 URL 已返回 404，改用仍可访问的 PyTorch 官方博客版本。
- TVM 旧的 `tensor_ir/tutorials/tutorial.html` 已返回 404，改用 `tir_creation.html` 与 `tir_transformation.html`。
- 旧的 PyTorch `torchdynamo_deepdive.html` 已返回 404，改用仍在维护的 `torch_compile_full_example.html` 和 Core Concepts 文档。

网站页面会继续变化。正式写入正文后，应在 CI 中增加外链检查，但对 403、限流和需要 JavaScript 的站点设置合理重试或 allowlist，避免把暂时性网络失败误判成死链。
