import json

class RedisManager:
    def __init__(self, redis_client=None, log_queue=None, heartbeat_queue=None, error_queue=None):
        if redis_client is None:
            raise ValueError("Redis client cannot be None")
        self.r = redis_client
        self.log_queue = log_queue
        self.heartbeat_queue = heartbeat_queue
        self.error_queue = error_queue
        self.running = False

    def send(self, queue_name: str, packet: dict) -> bool:
        try:
            json_data = json.dumps(packet)
            self.r.rpush(queue_name, json_data)
            return True
        except Exception as e:
            return False

    def receive(self, queue_name: str, timeout: int = 0) -> dict:
        try:
            # blpop -> (queue_name, item) şeklinde bir tuple döner
            result = self.r.blpop(queue_name, timeout=timeout)
            if result:
                return json.loads(result[1])
            return {}
        except Exception as e:
            return {}

    def distribute(self, source_queue: str, routing_map: dict, timeout: int = 0) -> bool:
        packet = self.receive(source_queue, timeout=timeout)
        
        if packet:
            packet_type = packet.get("type", "unknown")
            
            target_queue = routing_map.get(packet_type)
            
            if target_queue:
                self.send(target_queue, packet)
                return True
            else:
                return False
                
        return False

    def listen(self, queue_name: str = "manager_queue", timeout: int = 1):
        self.running = True
        while self.running:
            packet = self.receive(queue_name, timeout=timeout)
            if packet:
                packet_type = packet.get("type")
                if packet_type == "log":
                    # Error-flagged logs get routed to their own queue so
                    # Manager can react to them separately from normal info
                    # logs (see Manager.error_handler_thread).
                    if packet.get("errornum", 0) > 0 and self.error_queue is not None:
                        self.error_queue.put(packet)
                    elif self.log_queue is not None:
                        self.log_queue.put(packet)
                elif packet_type == "heartbeat" and self.heartbeat_queue is not None:
                    self.heartbeat_queue.put(packet)

    def stop_listening(self):
        self.running = False