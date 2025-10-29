# Ray Summit 2025

Developer Guide [developer_guide.md](developer_guide.md).

---
**Abstract**  
Processing large-scale image datasets for captioning presents coordination challenges that often lead to complex, difficult-to-maintain systems. I've been exploring how Ray Data can simplify these workflows while improving throughput and reliability.This talk demonstrates how to build image captioning pipelines combining Ray Data's batch processing capabilities,  Ray Data LLM's batch inference capabilities, vLLM for efficient model serving. I'll walk through how we:

- Structure data processing pipelines using Ray Data's map and map_batches operations
- Use Ray Data LLM's [Processor](https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.Processor.html#ray.data.llm.Processor) object which encapsulates logic for performing batch inference with LLMs on a Ray Data dataset.
- Integrate vLLM for high-throughput batch inference on vision-language models
- Handle fault tolerance and checkpointing for long-running jobs
- Optimize GPU resource utilization across distributed workloads

We'll explore practical patterns for processing thousand of images, including data loading strategies, batching considerations, and state management approaches. The talk showcases how Ray Data's and Ray Data LLM's abstractions can replace complex actor coordination patterns, demonstrating a path from prototype-scale scripts to production-ready pipelines that can handle real-world computer vision datasets.

**Target Audience**  
ML engineers working with large-scale vision datasets, researchers scaling computer vision experiments, and teams building production ML pipelines.

**Talk Outline (30 minutes)**  

**Problem Context (8 minutes)**  
+ Challenges in large-scale image processing workflows
+ Common patterns: actor coordination, queue management, state tracking
+ Trade-offs between simplicity and scale in existing approaches

**Ray Data Approach (15 minutes)**  
+ Ray Data fundamentals for batch processing
+ Integration patterns with vLLM for vision-language models
+ Code walkthrough: data loading, batching, and result handling
+ Resource management and GPU sharing strategies

**Production Considerations (5 minutes)**
+ Checkpointing and restart strategies
+ Error handling and monitoring approaches
+ Performance characteristics and optimization techniques

**Q&A (2 minutes)**  
**Key Technical Points**
+ Practical Ray Data usage patterns for vision workloads
+ vLLM integration for efficient batch inference
+ Resource optimization techniques for GPU-intensive pipelines
+ State management without external coordination systems
