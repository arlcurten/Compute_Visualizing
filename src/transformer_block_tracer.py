"""
Description:
    This script simulates the forward pass of a transformer block in a language model (LLAMA) 
    and generates a trace of the operations performed.
"""

import torch
import os
from llama_loader import load_llama_model
from scheduler import Scheduler, generate_trace_events
from perfetto_writer import PerfettoTraceWriter
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import time

def simulate_transformer_block(block, config, device, Total_TokenCount):
    # Create dummy inputs for decoding (autoregressive, single token)
    D_q = block.self_attn.q_proj.in_features
    D_k = block.self_attn.k_proj.in_features
    D_v = block.self_attn.v_proj.in_features

    N = 4  # Cached KV length (number of key-value pairs in the cache)
    #D_q = D_k = D_v = 16
    q = torch.randn(1, 1, D_q).to(device)  # Query vector (batch=1, seq_len=1, dim=D_q)
    k = torch.randn(1, N, D_k).to(device)  # Key (batch=1, seq_len=N, dim=D_k)
    v = torch.randn(1, N, D_v).to(device)  # Value (batch=1, seq_len=N, dim=D_v)

    # Simulate the forward pass through the transformer block to trace the operations
    ops = []

    for TokenCount in range(1, Total_TokenCount+1):
        # Simulate the operations in the transformer block
        # For each operation, we will record its name, type, inputs, outputs, duration, and size
        # The duration is simulated based on the operation type and data size

        # 1. Memory Transfer: Loading KV Cache (keys and values)
        ops.append({"name": "mem_transfer_load_kv_cache", "type": "mem_transfer_load_kv_cache", "inputs": ["kv_cache"], "output": ["k_load", "v_load"], "dur": 1200, "output_size": [list(k.shape), list(v.shape)], "token_n_count": TokenCount})

        # 2. LayerNorm before attention
        start = time.perf_counter()
        norm_q = block.input_layernorm(q)
        dur = (time.perf_counter() - start) * 1e6
        ops.append({"name": "LayerNorm1", "type": "layernorm", "inputs": ["out"], "output": "norm_q", "dur": dur, "output_size": list(norm_q.shape), "token_n_count": TokenCount})

        # 3. Project QKV
        start = time.perf_counter()
        q_proj = block.self_attn.q_proj(norm_q)  # [batch_size, seq_len, D_q]
        dur_q = (time.perf_counter() - start) * 1e6
        ops.append({"name": "Q_proj", "type": "linear", "inputs": ["norm_q"], "output": "q_proj", "dur": dur_q, "output_size": list(q_proj.shape), "token_n_count": TokenCount})

        start = time.perf_counter()
        k_proj = block.self_attn.k_proj(k)
        if k_proj.shape[-1] != q_proj.shape[-1]:
            projection_layer = torch.nn.Linear(k_proj.shape[-1], q_proj.shape[-1]).to(device)
            k_proj = projection_layer(k_proj)  # manually project k_proj to match q_proj
        dur_k = (time.perf_counter() - start) * 1e6
        ops.append({"name": "K_proj", "type": "linear", "inputs": ["out"], "output": "k_proj", "dur": dur_k, "output_size": list(k_proj.shape), "token_n_count": TokenCount})

        start = time.perf_counter()
        v_proj = block.self_attn.v_proj(v)  # [batch_size, seq_len, D_v]
        if v_proj.shape[-1] != q_proj.shape[-1]:
            projection_layer2 = torch.nn.Linear(v_proj.shape[-1], q_proj.shape[-1]).to(device)
            v_proj = projection_layer2(v_proj)  # manually project v_proj to match q_proj
        dur_v = (time.perf_counter() - start) * 1e6
        ops.append({"name": "V_proj", "type": "linear", "inputs": ["out"], "output": "v_proj", "dur": dur_v, "output_size": list(v_proj.shape), "token_n_count": TokenCount})
        
        # Reshape QKV for multi-head attention
        seq_len_q = 1
        seq_len_k = N
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        q_proj = q_proj.view(1, seq_len_q, num_heads, head_dim).transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        k_proj = k_proj.view(1, seq_len_k, num_heads, head_dim).transpose(1, 2)  # where head_dim = hidden_size / num_attention_heads
        v_proj = v_proj.view(1, seq_len_k, num_heads, head_dim).transpose(1, 2)

        # 4. Apply simplified RoPE manually (TBD for multi-head)
        cos_q = torch.ones_like(q_proj)
        sin_q = torch.zeros_like(q_proj)
        cos_k = torch.ones_like(k_proj)
        sin_k = torch.zeros_like(k_proj)

        start = time.perf_counter()
        q_rope, _ = apply_rotary_pos_emb(q_proj, cos_q, sin_q, sin_q)
        dur = (time.perf_counter() - start) * 1e6
        ops.append({"name": "RoPE_q", "type": "rotary_embedding", "inputs": ["q_proj"], "output": "q_rope", "dur": dur, "output_size": list(q_rope.shape), "token_n_count": TokenCount})

        start = time.perf_counter()
        k_rope, _ = apply_rotary_pos_emb(k_proj, cos_k, sin_k, sin_k)
        dur = (time.perf_counter() - start) * 1e6
        ops.append({"name": "RoPE_k", "type": "rotary_embedding", "inputs": ["k_proj"], "output": "k_rope", "dur": dur, "output_size": list(k_rope.shape), "token_n_count": TokenCount})

        
        # 4-2. Memory Transfer: Storing updated KV cache
        ops.append({"name": "mem_transfer_store_kv_cache", "type": "mem_transfer_store_kv_cache", "inputs": ["k_rope", "v_proj", "v_load", "k_load"], "output": "kv_cache", "dur": 1200, "output_size": [list(k.shape), list(v.shape)], "token_n_count": TokenCount})


        # 5-1. Attention computation
        k_rope_T = k_rope.transpose(-2, -1)  # [1, num_heads, head_dim, seq_len_k]
        start = time.perf_counter()
        scores = torch.matmul(q_rope, k_rope_T) / (head_dim ** 0.5)  # [1, num_heads, seq_len_q, seq_len_k]
        dur_scores = (time.perf_counter() - start) * 1e6
        for head in range(num_heads):  # output dimensions TBD
            ops.append({"name": "QK^T", "type": "multihead_dot", "inputs": ["q_rope", "k_rope", "k_read"], "output": "scores", "dur": dur_scores/num_heads, "output_size": list(scores.shape), "token_n_count": TokenCount, "head_cnt": head})

        # 5-2. Softmax attention weights
        start = time.perf_counter()
        max_scores = torch.max(scores, dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores)
        attn_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
        dur_softmax = (time.perf_counter() - start) * 1e6  # divide by 3 for each softmax procedure below
        dur_softmax_max = dur_softmax / 3
        dur_softmax_exp = dur_softmax / 3
        dur_softmax_norm = dur_softmax / 3
        for head in range(num_heads):  # output dimensions TBD
            ops.append({"name": "softmax_max", "type": "softmax_max", "inputs": ["scores"], "output": "max_scores", "size": (1, N), "dur": dur_softmax_max / num_heads, "output_size": list(max_scores.shape), "token_n_count": TokenCount, "head_cnt": head})
            ops.append({"name": "softmax_exp", "type": "softmax_exp", "inputs": ["scores", "max_scores"], "output": "exp_scores", "size": (1, N), "dur": dur_softmax_exp / num_heads, "output_size": list(exp_scores.shape), "token_n_count": TokenCount, "head_cnt": head})
            ops.append({"name": "softmax_norm", "type": "softmax_norm", "inputs": ["exp_scores"], "output": "attn_weights", "size": (1, N), "dur": dur_softmax_norm / num_heads, "output_size": list(attn_weights.shape), "token_n_count": TokenCount, "head_cnt": head})

        # 5-3. Weighted sum of values
        start = time.perf_counter()
        context = torch.matmul(attn_weights, v_proj)  # [1, num_heads, seq_len_q, head_dim]
        dur_context = (time.perf_counter() - start) * 1e6
        for head in range(num_heads):  # output dimensions TBD
            ops.append({"name": "Attn x V", "type": "multihead_dot", "inputs": ["attn_weights", "v_proj", "v_load"], "output": "context", "dur": dur_context/num_heads, "output_size": list(context.shape), "token_n_count": TokenCount, "head_cnt": head})
        
        # 6. Merge heads & output projection
        context = context.transpose(1, 2).contiguous().view(1, seq_len_q, -1)  # [1, seq_len_q, hidden_size]
        attn_out = block.self_attn.o_proj(context)								 
        ops.append({"name": "Output projection", "type": "linear", "inputs": ["context"], "output": "attn_out", "dur": 100.0, "output_size": list(attn_out.shape), "token_n_count": TokenCount})

        # 7. Residual connection after attention
        start = time.perf_counter()
        attn_residual = attn_out = context + q
        ops.append({"name": "Residual1", "type": "add", "inputs": ["attn_out", "q"], "output": "attn_residual", "dur": (time.perf_counter() - start) * 1e6, "output_size": list(attn_residual.shape), "token_n_count": TokenCount})

        # 8. LayerNorm before MLP
        start = time.perf_counter()
        mlp_input = block.post_attention_layernorm(attn_residual)
        dur = (time.perf_counter() - start) * 1e6
        ops.append({"name": "LayerNorm2", "type": "layernorm", "inputs": ["attn_residual"], "output": "mlp_input", "dur": dur, "output_size": list(mlp_input.shape), "token_n_count": TokenCount})

        # 9. MLP
        start = time.perf_counter()
        mlp_out = block.mlp(mlp_input)
        dur = (time.perf_counter() - start) * 1e6
        ops.append({"name": "MLP", "type": "dot", "inputs": ["mlp_input"], "output": "mlp_out", "dur": dur, "output_size": list(mlp_out.shape), "token_n_count": TokenCount})

        # 10. Add & Normalize
        start = time.perf_counter()
        out = mlp_out + mlp_input
        out = block.post_attention_layernorm(out)
        ops.append({"name": "Residual2", "type": "add", "inputs": ["mlp_input", "mlp_out"], "output": "out", "dur": (time.perf_counter() - start) * 1e6, "output_size": list(out.shape), "token_n_count": TokenCount})
            # little trick to put "mlp_out" after "mlp_input" in the inputs list so that scheduling will keep it in the same engine of MLP

    return ops, num_heads



def main():
    #------Load model and prepare block------#
    device = 'cuda'  # if supported
    # device = 'cpu'  # default device
    model, config, tokenizer = load_llama_model(device=device)  # load the entire model

    # Extract one transformer block
    block = model.model.layers[0]  # For instance, the first transformer block (LLAMA architecture)

    #------dump the model into a text for self reference------#
    output_dir = "outputs"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    filename_dumpModel = os.path.join(output_dir, "llama_model_structure.txt")
    with open(filename_dumpModel, "w") as f:
        print(model, file=f)
        print("\n\n", file=f)

    # dump the block into a text for self reference
    with open(filename_dumpModel, "a") as f:
        print(block, file=f)
    print("Dumped the model structure into: ", filename_dumpModel)


    #------Simulate the forward pass and trace operations------#
    n_threads = 4  # Number of threads (engines) to simulate
    Total_TokenCount = 2  # Number of tokens in the input sequence
    ops, num_heads = simulate_transformer_block(block, config, device, Total_TokenCount)


    #------Schedule the ops across threads------#
    scheduler = Scheduler(num_engines=n_threads)
    trace_events = generate_trace_events(ops, scheduler, num_heads, Total_TokenCount)

    #------Write the trace events to a Perfetto trace file------#
    filename_erfettoTrace = os.path.join(output_dir, "transformer_trace.json")
    writer = PerfettoTraceWriter(filename_erfettoTrace)
    for event in trace_events:
        writer.add_event(
            name=event["name"],
            ts=event["ts"],
            dur=event["dur"],
            thread_id=event["thread_id"],
            args=event["args"]
        )
    writer.write()
    print("Perfetto trace written into: ", filename_erfettoTrace)



if __name__ == "__main__":
    main()
