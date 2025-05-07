# Compute_visualizing

**Purpose**: Visualizing LLAMA3.2 Compute Chain

**Compute Operations**: dot product, 3-pass softmax, normalization, positional encoding, memory transfers

**Settings**: 
* [LLAMA-3.2-1B model](https://huggingface.co/meta-llama/Llama-3.2-1B) from HuggingFace
* Auto regressive decoding
* One transformer block
* [Perfetto](https://ui.perfetto.dev/) trace for visualization
* Dynamically adapts to changes
  
**Relative parameters**:
* Query matrix $Q$: shape $(1, D)$
* Cached key matrix $K$: shape $(N, D)$
* Cached value matrix $V$: shape $(N, D)$
* Dimensions $N=4$ (KV size) and $D=16$ (engine_size)
* 4 concurrent compute engines

**Results**:
* Without parallelism
![without parallelism.jpg](/no_parallelism.jpg)
<br/>

* With parallelism(TBD)

<br/>


# Project Structure
* (TBD)

**Additional Information**: 
* customerized Attention
* 3-pass softmax
* num3 projection num4 RoPE are modified and no the exact



**To-do Items**:
1. torch.profile → automatically update?

<br/>

**References**:
1. [Exploring and building the LLaMA 3 Architecture : A Deep Dive into Components, Coding, and Inference Techniques](https://medium.com/@vi.ai_/exploring-and-building-the-llama-3-architecture-a-deep-dive-into-components-coding-and-43d4097cfbbb)
2. ChatGPT
