---
title: "第7章：Prefill/Decode 解耦架构"
description: "从两阶段互扰到 KV 传输、SLO 路由、容量配比和 vLLM 实验"
pubDate: 2026-10-03
category: "inference-optimization"
order: 36
tags: ["Disaggregated Prefill", "PD 解耦", "KV Transfer", "Goodput", "vLLM"]
---

Prefill 与 Decode 使用同一模型，却有不同的计算形状和延迟目标。P/D 解耦把两阶段放入独立 GPU 池，通过传输 KV Cache 衔接请求。它解决资源互扰，也引入网络、路由、容量和失败恢复的新成本。

## 主线教程

[AWS：Disaggregated Prefill and Decode for LLM Inference](https://aws.amazon.com/cn/blogs/machine-learning/disaggregated-prefill-and-decode-for-llm-inference-on-sagemaker-hyperpod/) 是本章主线：从动机、架构、EFA/RDMA 到部署步骤都有图和配置。先通读，再按下列小节补齐通用概念。

## 本章小节

- **7.1 混合 Batching 与解耦动机**：什么时候值得拆，什么时候统一引擎反而更好。
- **7.2 KV 传输与 Connector**：一次请求如何跨池，以及不同项目中的术语对应关系。
- **7.3 Goodput 与 SLO 路由**：从 TTFT/TPOT 约束推导路由决策。
- **7.4 资源配比与工程挑战**：用到达率、服务时间和 KV 流量估算 P:D。
- **7.5 vLLM 解耦实验**：按基线、传输、故障和端到端 SLO 验证。

<figure class="source-figure">
  <img src="/AIInfraGuide/images/inference-optimization/dynamo-disaggregated-serving.svg" alt="NVIDIA Dynamo 的 Prefill 和 Decode 解耦服务架构" />
  <figcaption>NVIDIA Dynamo 的解耦服务架构图，展示独立 Prefill/Decode worker、路由和 KV 数据路径。<a href="https://github.com/ai-dynamo/dynamo">图源</a>（Apache-2.0）</figcaption>
</figure>
