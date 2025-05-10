"""
Description:
    This module provides functionality to write performance tracing data in the Perfetto format.

    It includes a `PerfettoTraceWriter` class to create and write trace events to a JSON file,
    and a function `generate_trace_events` to generate trace events based on operations and a scheduler.
"""

import json

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
