# profiler.py

import random

def estimate_duration(op_type: str, size=None):
    """
    Simulates operation durations (in microseconds) based on operation type and data size.
    These values are heuristic and not tied to real hardware.
    """
    base_durations = {
    "layernorm": 300,
    "linear": 800,
    "rotary_embedding": 500,
    "dot": 1000,
    "softmax_max": 300,
    "softmax_exp": 400,
    "softmax_sum": 300,
    "softmax_norm": 400,
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
    if size is not None:
        # Rough scaling: larger tensors take more time
        scale = 1 + 0.01 * sum(size)
        duration = int(duration * scale)

    # Add some noise for realism
    return duration + random.randint(-50, 50)
