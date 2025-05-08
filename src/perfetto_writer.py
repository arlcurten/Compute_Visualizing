"""
This module provides functionality to write performance tracing data in the Perfetto format.

It includes a `PerfettoTraceWriter` class to create and write trace events to a JSON file,
and a function `generate_trace_events` to generate trace events based on operations and a scheduler.
"""

import json
from profiler import estimate_duration

class PerfettoTraceWriter:
    def __init__(self, filename):
        self.filename = filename
        self.trace_events = []

    def add_event(self, name, ts, dur, thread_id, args):
        self.trace_events.append({
            "name": name,
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": 0,
            "tid": thread_id,
            "args": args
        })

    def write(self):
        with open(self.filename, 'w') as f:
            json.dump({"traceEvents": self.trace_events}, f, indent=2)

def generate_trace_events(ops, scheduler):
    trace_events = []
    for op in ops:
        op_type = op["type"]
        duration = estimate_duration(op_type)
        engine_id, start_time = scheduler.schedule(duration)

        if op_type == "mem_transfer":
            # shape,  = op["shape"]
            # description = op["description"]
            output_size = op["output_size"]
            trace_events.append({
                "name": f"mem_transfer: {op_type}",
                "ts": start_time,
                "dur": duration,
                "thread_id": engine_id,
                "args": {
                    "type": "mem_transfer",
                    #"description": description,
                    #"input_shape": shape,
                    "output_shape": output_size,
                }
            })
        else:
            # shape = op["shape"]
            output_size = op["output_size"]
            trace_events.append({
                "name": op_type,
                "ts": start_time,
                "dur": duration,
                "thread_id": engine_id,
                "args": {
                    "type": op_type,
                    #"input_shape": shape,
                    "output_shape": output_size,
                }
            })
    return trace_events
