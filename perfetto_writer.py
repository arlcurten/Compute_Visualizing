# perfetto_writer.py
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
            trace_events.append({
                "name": f"mem_transfer: {op_type}",
                "ts": start_time,
                "dur": duration,
                "thread_id": engine_id,
                "args": {
                    "type": "mem_transfer",
                    #"description": description,
                    #"input_shape": shape,
                    #"output_shape": shape,
                }
            })
        else:
            # shape = op["shape"]
            trace_events.append({
                "name": op_type,
                "ts": start_time,
                "dur": duration,
                "thread_id": engine_id,
                "args": {
                    "type": op_type,
                    #"input_shape": shape,
                    #"output_shape": shape,
                }
            })
    return trace_events
