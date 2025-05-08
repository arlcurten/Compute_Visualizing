"""
Description: 
    Round Robin Scheduler (for testing)
    Here, each task is globally blocking (i.e., no parallelism at all)
"""

class RoundRobinScheduler:
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
TBD a real scheduler for optimizing the scheduling of tasks across multiple engines
"""
"""
class RoundRobinScheduler:
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
"""