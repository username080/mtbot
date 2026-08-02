import time

class HeartbeatTracker:
    def __init__(self):
        self.last_heartbeats = {}

    def record_heartbeat(self, worker_name: str, timestamp: str = "Unknown"):
        ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_heartbeats[worker_name] = ts
        print(f"[HEARTBEAT] Worker '{worker_name}' is alive (recorded at {ts})")

    def get_status(self) -> dict:
        return self.last_heartbeats
