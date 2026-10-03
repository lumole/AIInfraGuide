---
title: "第9章：性能分析与 Benchmark"
description: "从指标定义、压测协议到 Kernel 定位、标准基准和回归门禁"
pubDate: 2026-10-03
category: "inference-optimization"
order: 38
tags: ["Benchmark", "TTFT", "TPOT", "Profiler", "MLPerf"]
---

性能数字只有在负载、指标和环境都定义清楚时才有意义。本章按“先定义、再测量、后定位、最后门禁”的顺序拆成五节。

## 本章小节

- **9.1 指标体系与 SLO**：TTFT、TPOT、ITL、吞吐与 goodput 的准确关系。
- **9.2 压测协议与工具**：open/closed loop、长度分布、warm-up 与可复现 manifest。
- **9.3 从服务指标到 Kernel**：把请求级回归逐层定位到调度、CPU、通信或 GPU kernel。
- **9.4 MLPerf 与权威基准**：学会读规则和结果，不把公开榜单误当业务结论。
- **9.5 性能回归门禁**：处理噪声、阈值、重复实验和自动 bisect。

<figure class="source-figure">
  <img src="/AIInfraGuide/images/inference-optimization/vllm-latency-intervals.png" alt="LLM 请求中 TTFT、ITL 和端到端延迟的时间区间" />
  <figcaption>vLLM 指标示意图，用于统一 TTFT、ITL 与端到端延迟的时间边界。<a href="https://github.com/vllm-project/vllm">图源</a>（Apache-2.0）</figcaption>
</figure>
