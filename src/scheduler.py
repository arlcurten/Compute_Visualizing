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
        self.map_output_to_time = {}  # key:output_name, value:[end_time, engine_id]
        # self.current_engine = 0  # Tracks the next engine to assign work to

    def schedule(self, duration, op):
        # Search "end_time" of the output_name which is the input of the current op
        start_time = 0
        for input in op["inputs"]: 
            if input in self.map_output_to_time:  # Check if the input is an output of a previous operation
                pre_end_time, pre_engine_id = self.map_output_to_time[input]
                start_time = max(start_time, pre_end_time)  # select the max end_time of all inputs

        # Find the available engine at the earliest time
        engine_id = 0
        min_time = self.engine_avai_time[engine_id]  # start searching from engine=0
        for i in range(self.num_engines):
            if self.engine_avai_time[i] < min_time: 
                min_time = self.engine_avai_time[i]
                engine_id = i

        # The start time of the current op is the max of the engine's available time and the start_time of the input
        start_time = max(start_time, min_time)  # The start time of the current op is the max of the engine's available time and the start_time of the input
        
        # Update the engine's available time, fitted to multi-head condition
        self.engine_avai_time[engine_id] = start_time + duration    
        
        # Update the map_output_to_time
        # In multi-head condition, the output name is the same for all heads; hence, it will be updated the last (max) one
        if op["name"] == "mem_transfer_load_kv_cache":  # special case with 2 outputs for mem_transfer_load_kv_cache
            self.map_output_to_time[op["output"][0]] = [self.engine_avai_time[engine_id], engine_id]
            self.map_output_to_time[op["output"][1]] = [self.engine_avai_time[engine_id], engine_id]
        else:
            self.map_output_to_time[op["output"]] = [self.engine_avai_time[engine_id], engine_id] 

        return engine_id, start_time



def generate_trace_events(ops, scheduler, num_heads=1, Total_TokenCount=1):
    trace_events = []
    for op in ops:
        op_type = op["type"]
        duration = estimate_duration(op_type, num_heads)   # mapping op_type to estimated duration in profiler.py
        # duration = op["dur"]  # Use the duration from the operation if available
        engine_id, start_time = scheduler.schedule(duration, op)
        TokenCount = op["toke_n_count"] if "toke_n_count" in op else 1
        head_cnt = op["head_cnt"] if "head_cnt" in op else 0

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
                    "toke_count": TokenCount,
                    "head_count": head_cnt
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
                    "toke_count": TokenCount,
                    "head_count": head_cnt
                }
            })
    print(f"Finished scheduling operations for {scheduler.num_engines} threads, {num_heads} heads, {Total_TokenCount} tokens")
    return trace_events
