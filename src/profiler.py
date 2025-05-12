"""
Description:
    This module provides a function to estimate the duration of various operations in a transformer model.
    The durations are simulated based on operation type and data size, with some added randomness for realism.

    The estimated durations are not tied to real hardware and are intended for profiling purposes only.
    An ideal case would involve using actual hardware performance counters or profiling tools to get accurate timings.
    For example: using NVIDIA's nvprof or PyTorch's built-in profiler to acquire real information from forward pass operations.
"""

import random

def estimate_duration(op_type: str, num_heads=1):
    """
    Simulates operation durations (in microseconds) based on operation type and data size.
    These values are heuristic and not tied to real hardware.
    """
    base_durations = {
    "layernorm": 300,
    "linear": 800,
    "rotary_embedding": 500,
    "multihead_dot": 1000/num_heads,
    "dot": 1000,
    "softmax_max": 300/num_heads,
    "softmax_exp": 400/num_heads,
    "softmax_norm": 400/num_heads,
    # "softmax_sum": 300/num_heads,
    "mask": 200,
    "add": 300,
    "reduce_sum": 300,
    "mlp": 1200,
    
    # Estimated durations for different memory transfer scenarios
    "mem_transfer_load_kv_cache": 1200,  # Loading keys/values (e.g., for attention) into memory
    "mem_transfer_store_kv_cache": 1500,  # Storing keys/values back to memory
    # "mem_transfer_device_to_device": 2000,  # Data transfer between devices (e.g., GPU ↔ CPU)
    # "mem_transfer_device_to_host": 1500,  # GPU → CPU memory transfer
    # "mem_transfer_host_to_device": 1800,  # CPU → GPU memory transfer
    }

    # Adjust duration based on size (if available)
    duration = base_durations[op_type]

    # Rough scaling: larger tensors take more time
    scale = 1 + 0.01
    duration = int(duration * scale)
    # Add some noise for realism (+-5%)
    duration += random.randint(-int(duration*0.05), int(duration*0.05))

    return duration
