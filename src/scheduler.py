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

    def schedule(self, duration, op):
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
        self.engine_avai_time = [0] * num_engines  # Tracks next available time for each engine
        self.map_output_to_time = {}  # key:output, value:[end_time, engine_id]
        # self.current_engine = 0  # Tracks the next engine to assign work to

    def schedule(self, duration, op):
        # search "end_time" of the output which is the input of the current op
        start_time = 0
        for input in op["inputs"]: 
            if input in self.map_output_to_time:  # Check if the input is an output of a previous operation
                pre_end_time, pre_engine_id = self.map_output_to_time[input]
                start_time = max(start_time, pre_end_time)
        # find the engine which is available at start_time
        engine_id = 0
        for i in range(self.num_engines):
            if self.engine_avai_time[i] <= start_time:
                engine_id = i
                break
        # update the engine's available time
        self.engine_avai_time[engine_id] = start_time + duration    
        # update the map_output_to_time
        if op["name"] == "mem_transfer_load_kv_cache":  # special case for mem_transfer_load_kv_cache
            self.map_output_to_time[op["output"][0]] = [self.engine_avai_time[engine_id], engine_id]
            self.map_output_to_time[op["output"][1]] = [self.engine_avai_time[engine_id], engine_id]
        else:
            self.map_output_to_time[op["output"]] = [self.engine_avai_time[engine_id], engine_id] 

        return engine_id, start_time



def generate_trace_events(ops, scheduler):
    trace_events = []
    for op in ops:
        op_type = op["type"]
        duration = estimate_duration(op_type)   # mapping op_type to estimated duration in profiler.py
        # duration = op["dur"]  # Use the duration from the operation if available
        engine_id, start_time = scheduler.schedule(duration, op)

        if op_type == "mem_transfer":
            # shape,  = op["shape"]
            # description = op["description"]
            output_size = op["output_size"]
            trace_events.append({
                # "name": f"mem_transfer: {op_type}",
                "name":  op["name"],
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
                "name": op["name"],
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
