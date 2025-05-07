# transformer_block_tracer.py
import torch
from llama_loader import load_llama_model
from profiler import estimate_duration
from scheduler import simulate_block_schedule, RoundRobinScheduler
from perfetto_writer import PerfettoTraceWriter, generate_trace_events
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import time

def main():
    # Load model and prepare block
    # device = 'cpu'  # or 'cuda' if supported
    device = 'cuda'  # or 'cuda' if supported
    model, config, tokenizer = load_llama_model(device=device)

    # dump the model into a text for self reference
    with open("llama_model_structure.txt", "w") as f:
        print(model, file=f)
        print("\n\n", file=f)
    
    # Extract one transformer block
    block = model.model.layers[0]  # For instance, the first transformer block (LLAMA architecture)

    # dump the block into a text for self reference
    with open("llama_model_structure.txt", "a") as f:
        print(block, file=f)

    # Create dummy inputs for decoding (autoregressive, single token)
    D_q = block.self_attn.q_proj.in_features
    D_k = block.self_attn.k_proj.in_features
    D_v = block.self_attn.v_proj.in_features

    N = 4  # Cached KV length (number of key-value pairs in the cache)
    q = torch.randn(1, 1, D_q).to(device)  # Query vector (batch=1, seq_len=1, dim=D_q)
    k = torch.randn(1, N, D_k).to(device)  # Key (batch=1, seq_len=N, dim=D_k)
    v = torch.randn(1, N, D_v).to(device)  # Value (batch=1, seq_len=N, dim=D_v)

    # Simulate the forward pass through the transformer block to trace the operations
    ops = []



    # 1. LayerNorm before attention
    start = time.perf_counter()
    norm_q = block.input_layernorm(q)
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "LayerNorm1", "type": "layernorm", "inputs": ["q"], "output": "norm_q", "dur": dur})

    # 2. Project queries and keys
    start = time.perf_counter()
    q_proj = block.self_attn.q_proj(norm_q)
    dur_q = (time.perf_counter() - start) * 1e6
    ops.append({"name": "Q_proj", "type": "linear", "inputs": ["norm_q"], "output": "q_proj", "dur": dur_q})

    start = time.perf_counter()
    k_proj = block.self_attn.k_proj(k)
    if k_proj.shape[-1] != q_proj.shape[-1]:
        projection_layer = torch.nn.Linear(k_proj.shape[-1], q_proj.shape[-1]).to(device)
        k_proj = projection_layer(k_proj)
    dur_k = (time.perf_counter() - start) * 1e6
    ops.append({"name": "K_proj", "type": "linear", "inputs": ["k"], "output": "k_proj", "dur": dur_k})

    # 3. Apply simplified RoPE manually
    seq_len_q = 1
    seq_len_k = N
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    q_proj = q_proj.view(1, seq_len_q, num_heads, head_dim)
    k_proj = k_proj.view(1, seq_len_k, num_heads, head_dim)
    cos_q = torch.ones_like(q_proj)
    sin_q = torch.zeros_like(q_proj)
    cos_k = torch.ones_like(k_proj)
    sin_k = torch.zeros_like(k_proj)

    start = time.perf_counter()
    q_rope, _ = apply_rotary_pos_emb(q_proj, cos_q, sin_q, sin_q)
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "RoPE_q", "type": "rotary_embedding", "inputs": ["q_proj"], "output": "q_rope", "dur": dur})

    start = time.perf_counter()
    k_rope, _ = apply_rotary_pos_emb(k_proj, cos_k, sin_k, sin_k)
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "RoPE_k", "type": "rotary_embedding", "inputs": ["k_proj"], "output": "k_rope", "dur": dur})

    q_rope = q_rope.view(1, seq_len_q, -1)
    k_rope = k_rope.view(1, seq_len_k, -1)
    min_dim = min(q_rope.shape[-1], k_rope.shape[-1])
    q_rope = q_rope[:, :, :min_dim]
    k_rope = k_rope[:, :, :min_dim]

    # 4. Attention computation
    k_rope_T = k_rope.transpose(-2, -1)
    start = time.perf_counter()
    scores = torch.matmul(q_rope, k_rope_T) / torch.sqrt(torch.tensor(min_dim, dtype=torch.float32))
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "QK^T", "type": "dot", "inputs": ["q_rope", "k^T"], "output": "scores", "size": (1, N), "dur": dur})

    start = time.perf_counter()
    max_scores = torch.max(scores, dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_scores)
    attn_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "softmax_max", "type": "softmax_max", "inputs": ["scores"], "output": "max_scores", "size": (1, N), "dur": dur / 3})
    ops.append({"name": "softmax_exp", "type": "softmax_exp", "inputs": ["scores", "max_scores"], "output": "exp_scores", "size": (1, N), "dur": dur / 3})
    ops.append({"name": "softmax_norm", "type": "softmax_norm", "inputs": ["exp_scores"], "output": "attn_weights", "size": (1, N), "dur": dur / 3})

    start = time.perf_counter()
    context = torch.matmul(attn_weights, v)
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "Attn x V", "type": "dot", "inputs": ["attn_weights", "v"], "output": "attn_out", "size": (1, D_q), "dur": dur})

    # 5. Residual connection after attention
    start = time.perf_counter()
    attn_residual = attn_out = context + q
    ops.append({"name": "Residual1", "type": "add", "inputs": ["attn_out", "q"], "output": "attn_residual", "dur": (time.perf_counter() - start) * 1e6})

    # 6. LayerNorm before MLP
    start = time.perf_counter()
    mlp_input = block.post_attention_layernorm(attn_residual)
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "LayerNorm2", "type": "layernorm", "inputs": ["attn_residual"], "output": "mlp_input", "dur": dur})

    # 7. MLP
    start = time.perf_counter()
    mlp_out = block.mlp(mlp_input)
    dur = (time.perf_counter() - start) * 1e6
    ops.append({"name": "MLP", "type": "dot", "inputs": ["mlp_input"], "output": "mlp_out", "dur": dur})

    # 8. Final residual connection
    start = time.perf_counter()
    final_output = mlp_out + attn_residual
    ops.append({"name": "Residual Final", "type": "add", "inputs": ["mlp_out", "attn_residual"], "output": "final_output", "dur": (time.perf_counter() - start) * 1e6})

    # Schedule the ops across threads
    scheduler = RoundRobinScheduler(num_engines=4)
    trace_events = generate_trace_events(ops, scheduler)

    writer = PerfettoTraceWriter("transformer_trace.json")
    for event in trace_events:
        writer.add_event(
            name=event["name"],
            ts=event["ts"],
            dur=event["dur"],
            thread_id=event["thread_id"],
            args=event["args"]
        )
    writer.write()
    print("Perfetto trace written to transformer_trace.json")

if __name__ == "__main__":
    main()