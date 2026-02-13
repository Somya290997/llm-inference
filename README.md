# Disaggregated LLM Inference System
### Prefill Decode Separation | KV Streaming | CPU Paging | Dynamic Decode Scheduling

A systems focused implementation of an LLM inference pipeline that separates prefill and decode execution across dual GPUs. This project explores how scheduling, memory movement, and resource coordination can improve latency and throughput beyond traditional single GPU serving approaches.

Built as part of Systems for Machine Learning coursework at the University of Colorado Boulder.

Authors  
Likhit Sai Kothapalli  
Somya Pathak  

---

## Motivation

Most work around LLMs focuses on model training or fine tuning. In real production environments, inference performance is often constrained by:

- GPU utilization inefficiencies  
- Memory transfer overhead  
- KV cache growth  
- Sequential decode bottlenecks  
- Queue latency under mixed workloads  

Prefill and decode have very different compute and memory characteristics. Running both on the same device leads to contention, idle cycles, and increased latency.

This project investigates system level optimization strategies that treat inference as a scheduling and memory management problem rather than purely a modeling problem.

---

## Key Ideas

### Prefill Decode Disaggregation
Prefill runs on one GPU while decode runs on another, allowing parallel execution and improved utilization.

### Layer Wise KV Streaming
KV cache is transferred layer by layer instead of waiting for full completion. This enables overlap between compute and communication.

### CPU Buffered KV Scheduling
KV data is staged in CPU pinned memory before transfer to decode GPU. This reduces interference with decode execution and allows smarter transfer scheduling.

### Dynamic Decode Optimization
- Bin based batching for similar sequence lengths  
- Immediate sequence eviction upon EOS  
- Adaptive batch composition  

### CPU Level Paging for KV Cache
Inspired by paged attention concepts.  
KV blocks are managed through CPU side paging structures when full GPU level implementation is constrained by architecture or framework limitations.

This improves memory footprint control and enables experimentation with space optimized KV management.

---

## Architecture Overview

Pipeline Components:

1. Input Queue  
2. Batch Worker  
3. Prefill Worker  
4. Transfer Scheduler  
5. CPU KV Buffer  
6. Transfer Worker  
7. Decode Worker  

High level flow:

Input → Batch Formation → Prefill GPU → CPU KV Paging Buffer → Scheduled Transfer → Decode GPU → Output

Core responsibilities:

- Prefill GPU generates KV representations  
- CPU buffer stages and tracks memory pages  
- Scheduler monitors decode GPU memory availability  
- Decode worker performs autoregressive generation with dynamic batching  

---

## Results

Test Environment  
Dual NVIDIA L4 GPUs  

Observed Performance Improvements

- 60% to 96% compute and transfer overlap  
- 20× reduction in Time To First Token  
- 4× throughput increase  

These results demonstrate the effectiveness of separating compute phases and coordinating memory movement.

---

## Repository Structure
