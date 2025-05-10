"""
Description: 
    Different type of scheduler implementation:
        Scheduler-Round Robin (for testing): each task is globally blocking (i.e., no parallelism at all)
        Scheduler: (TBD) a real scheduler for optimizing the scheduling of tasks across multiple engines

    It includes a `PerfettoTraceWriter` class to create and write trace events to a JSON file,
    and a function `generate_trace_events` to generate trace events based on operations and a scheduler.
"""

from profiler import estimate_duration


"""
Round Robin Scheduler (for testing)
    Here, each task is globally blocking (i.e., no parallelism at all)
"""
"""
class Scheduler:
    def __init__(self, num_engines=4):
        self.num_engines = num_engines
        self.current_engine = 0
        self.global_time = 0  # Enforces blocking across all engines

    def schedule(self, duration):
        engine_id = self.current_engine
        start_time = self.global_time  # Each task starts only after the last one ends
        self.global_time += duration   # Advance global timeline
        self.current_engine = (self.current_engine + 1) % self.num_engines
        return engine_id, start_time
"""


"""
TBD a real scheduler for optimizing the scheduling of tasks across multiple engines
"""
class Scheduler:
    def __init__(self, num_engines=4):
        self.num_engines = num_engines
        self.engines = [0] * num_engines  # Tracks next available time for each engine
        self.current_engine = 0  # Tracks the next engine to assign work to

    def schedule(self, duration):
        # Assign task to current engine
        engine_id = self.current_engine
        start_time = self.engines[engine_id]  # Start after this engine finishes its last task
        self.engines[engine_id] += duration   # Update this engine's timeline

        # Move to the next engine for the next task
        self.current_engine = (self.current_engine + 1) % self.num_engines

        return engine_id, start_time



def generate_trace_events(ops, scheduler):
    trace_events = []
    for op in ops:
        op_type = op["type"]
        duration = estimate_duration(op_type)   # mapping op_type to estimated duration in profiler.py
        # duration = op["dur"]  # Use the duration from the operation if available
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
