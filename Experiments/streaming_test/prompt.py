str1 = """This paper tackles the inefficiency that are present in current LLM serving systems that run the
prefill first-token generation and decoding (subsequent-token generation) phases on the same GPUs. This
causes underutilised resource and higher latency issue. The authors propose DistServe, a system that
disaggregates prefill and decoding onto separate GPUs, allowing each phase to be optimized independently. It
also introduces an algorithm that adapts GPU allocation and parallelism to maximize the number of requests that is
served within latency SLOs per GPU.The experiments on models can you summarise this for me please and also be short and clear ?"""

str2 = """This paper tackles the inefficiency that are present in current LLM serving systems that run the
prefill first-token generation and decoding (subsequent-token generation) phases on the same GPUs. This
causes underutilised resource and higher latency issue. The authors propose DistServe, a system that
disaggregates prefill and decoding onto separate GPUs, allowing each phase to be optimized independently. It
also introduces an algorithm that adapts GPU allocation and parallelism to maximize the number of requests that is
served within latency SLOs per GPU.The experiments on models can you summarise this for me please and also be short and clear ?"""


str3 = """This paper tackles the inefficiency that are present in current LLM serving systems that run the
prefill first-token generation and decoding (subsequent-token generation) phases on the same GPUs. This
causes underutilised resource and higher latency issue. The authors propose DistServe, a system that
disaggregates prefill and decoding onto separate GPUs, allowing each phase to be optimized independently. It
also introduces an algorithm that adapts GPU allocation and parallelism to maximize the number of requests that is
served within latency SLOs per GPU.The experiments on models can you summarise this for me please and also be short and clear ?"""