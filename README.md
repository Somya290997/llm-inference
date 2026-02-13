# Disaggregated LLM Inference System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![CUDA](https://img.shields.io/badge/CUDA-MultiGPU-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

### Prefill Decode Separation \| KV Streaming \| CPU Paging \| Dynamic Decode Scheduling

A systems focused implementation of an LLM inference pipeline that
separates prefill and decode execution across dual GPUs. This project
explores how scheduling, memory movement, and resource coordination can
improve latency and throughput beyond traditional single GPU serving
approaches.

Built as part of Systems for Machine Learning coursework at the
University of Colorado Boulder.

**Authors**\
Likhit Sai Kothapalli\
Somya Pathak

------------------------------------------------------------------------

## Architecture Diagram

``` text
                +------------------+
                |   Request Queue  |
                +---------+--------+
                          |
                          v
                +------------------+
                |    Scheduler     |
                +---------+--------+
                          |
        +-----------------+------------------+
        v                                    v
+------------------+              +------------------+
|   Prefill GPU    |              |  Decode GPU      |
| (FlashAttention) |              |  Autoregressive  |
+---------+--------+              +---------+--------+
          |                                 ^
          v                                 |
+-------------------------+                |
| CPU KV Paging Buffer    |----------------+
| Page Table + Scheduler  |
+-------------------------+
```

------------------------------------------------------------------------

## Motivation

Most work around LLMs focuses on model training or fine tuning. In real
production environments, inference performance is often constrained by:

-   GPU utilization inefficiencies\
-   Memory transfer overhead\
-   KV cache growth\
-   Sequential decode bottlenecks\
-   Queue latency under mixed workloads

Prefill and decode have very different compute and memory
characteristics. Running both on the same device leads to contention,
idle cycles, and increased latency.

This project investigates system level optimization strategies that
treat inference as a scheduling and memory management problem rather
than purely a modeling problem.

------------------------------------------------------------------------

## Key Ideas

### Prefill Decode Disaggregation

Prefill runs on one GPU while decode runs on another, allowing parallel
execution and improved utilization.

### Layer Wise KV Streaming

KV cache is transferred layer by layer instead of waiting for full
completion. This enables overlap between compute and communication.

### CPU Buffered KV Scheduling

KV data is staged in CPU pinned memory before transfer to decode GPU.
This reduces interference with decode execution and allows smarter
transfer scheduling.

### Dynamic Decode Optimization

-   Bin based batching for similar sequence lengths\
-   Immediate sequence eviction upon EOS\
-   Adaptive batch composition

### CPU Level Paging for KV Cache

Inspired by paged attention concepts. KV blocks are managed through CPU
side paging structures when full GPU level implementation was
constrained by architecture or framework limitations.

------------------------------------------------------------------------

## Results

**Test Environment**\
Dual NVIDIA L4 GPUs

  Metric                     Improvement
  -------------------------- -------------
  Compute Transfer Overlap   60% to 96%
  Time To First Token        20× faster
  Throughput                 4× higher

------------------------------------------------------------------------

## Demo Placeholder

Add GIF or screenshots here showing runtime execution or token streaming
output.

``` markdown
![Demo](docs/demo.gif)
```

------------------------------------------------------------------------

## Technologies Used

-   Python\
-   PyTorch\
-   CUDA\
-   Multi GPU execution\
-   FlashAttention2\
-   Hugging Face ecosystem

------------------------------------------------------------------------

## Takeaway

Efficient LLM serving is fundamentally a systems problem. Performance
gains often come from scheduling, memory management, and hardware
coordination rather than model architecture changes alone.
