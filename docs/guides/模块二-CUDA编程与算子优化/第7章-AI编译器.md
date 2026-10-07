---
title: "第7章：AI 编译器"
description: "掌握 Triton Block-level 编程模型、torch.compile 编译模式，以及 TVM/XLA 的定位与差异"
pubDate: 2026-04-16
category: "cuda-optimization"
order: 16
tags: ["Triton", "torch.compile", "AI编译器", "TorchInductor", "TVM"]
---

## 本章简介

手写 CUDA Kernel 可以获得很高的性能上限，但开发、调试和跨硬件维护成本也很高。AI 编译器通过更高层的抽象，把图捕获、算子融合、Tile 调度和代码生成的一部分交给编译器。本章从一个 Triton Kernel 开始，继续观察 `torch.compile` 的 Graph Break，最后建立 MLIR、TVM、XLA 的选型视角。

## 学习路径

| 小节 | 你会解决的问题 | 建议产出 |
| --- | --- | --- |
| [7.1 Triton 编程模型与 Fused Softmax](./71-triton编程模型与fused-softmax) | CUDA Thread 思维如何转为 Block-level program？ | 向量加法、Fused Softmax、shape autotune |
| [7.2 torch.compile 与 Inductor](./72-torchcompile与inductor实战) | 图捕获、融合、Graph Break 和编译缓存怎样观察？ | eager/compiled 对比与 break 日志 |
| [7.3 MLIR、TVM 与 XLA](./73-mlirtvm与xla选型) | 不同编译器处在哪一层，项目该如何选？ | 一张带约束的工具选型表 |

## 编译器学习的固定顺序

```text
先写 PyTorch 参考实现
        ↓
用 Triton 表达一个可控的 Block
        ↓
用 torch.compile 观察自动图捕获/融合
        ↓
查看生成 Kernel 和 Graph Break
        ↓
需要跨硬件或整图部署时再了解 TVM/MLIR/XLA
```

本章不把“编译成功”当成“性能更好”。每个实验都要同时记录正确性、首次编译时间、稳态延迟、Kernel 数、显存流量和 shape 变化时的重新编译。

## 章节完成标准

- 能用 Triton 写一个带 mask 的逐元素或 Softmax Kernel；
- 能解释 `program_id`、`tl.arange`、`tl.load/store` 和 `num_warps` 的含义；
- 能用 `TORCH_LOGS="graph_breaks,recompiles"` 定位编译图中断；
- 能区分 Triton、TorchInductor、TVM、XLA 和 MLIR 的抽象层；
- 能根据硬件、shape、部署生态和维护成本做出可解释的工具选择。

## 参考资料

- [Triton 官方 Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [PyTorch torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [MLIR 官方文档](https://mlir.llvm.org/docs/)
- [Apache TVM 官方文档](https://tvm.apache.org/docs/)
- [OpenXLA / XLA](https://openxla.org/xla)
