import os
import redis
import queue
import threading
import json
import socket
import os
from workers.analyser.analyser import Analyser
from workers.data_pipeline.fetcher import DataPipeline
from workers.executer.executer import Executer
from manager.logger_c import QuantLogger
from manager.heartbeat_c import HeartbeatTracker
from manager.redis_manager import RedisManager
from core.shm import shm_service
from core.services.binance_client_service import BinanceClientService
from core.packets.m_packet import MPacket

# ==========================================
# MANAGER CLASS
# ==========================================
class Manager:
    def __init__(self):
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.r = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.children = []
        
        # Queues for threading
        self.log_queue = queue.Queue()
        self.heartbeat_queue = queue.Queue()
        self.error_queue = queue.Queue()

        # Redis Manager
        self.redis_manager = RedisManager(
            redis_client=self.r,
            log_queue=self.log_queue,
            heartbeat_queue=self.heartbeat_queue,
            error_queue=self.error_queue,
        )
        
        # Logger setup
        self.logger_instance = QuantLogger()
        self.logger = self.logger_instance.get_logger()
        
        # Heartbeat tracker
        self.heartbeat_tracker = HeartbeatTracker()

        # Shutdown event for threads
        self.shutdown_event = threading.Event()

        # Initialize Shared Memory for portfolio using shm_service
        self.shm_name = shm_service.portfolio_name
        self.shm_lock = shm_service.portfolio_lock
        
        # Fetch initial account details from Binance (or fall back to dry-run default)
        binance_svc = BinanceClientService()
        balance = binance_svc.get_futures_balance()
        positions = binance_svc.get_open_positions()
        
        # Initial State Setup
        initial_state = {
            "portfolio": {"cash": balance, "positions": positions},
            "positions_info": {}
        }
        shm_service.init_shm(initial_state)

    # --- Logger Thread ---
    def logger_thread(self):
        """
        Handles normal (errornum == 0) log packets only - error-flagged
        ones never reach log_queue, RedisManager.listen() routes those to
        error_queue instead. See error_handler_thread.
        """
        print("Logger thread started.")
        while not self.shutdown_event.is_set():
            try:
                packet = self.log_queue.get(timeout=1.0)
                if packet is not None:
                    message = packet.get("msg", "")
                    timestamp = packet.get("timestamp", "")
                    sender = packet.get("sender", "unknown")

                    log_msg = f"[{sender}] [{timestamp}] {message}"
                    self.logger.info(log_msg)

                    self.log_queue.task_done()
            except queue.Empty:
                continue
        print("Logger thread shutting down...")

    # --- Error Handler Thread ---
    def error_handler_thread(self):
        """
        Handles error-flagged (errornum > 0) packets, kept separate from
        logger_thread so there's a dedicated place to decide how to react
        to a worker error - retry, alert, restart the worker, etc. No such
        policy is decided yet, so for now this just logs the error, same
        as logger_thread did for errors before they were split out.
        """
        print("Error handler thread started.")
        while not self.shutdown_event.is_set():
            try:
                packet = self.error_queue.get(timeout=1.0)
                if packet is not None:
                    message = packet.get("msg", "")
                    timestamp = packet.get("timestamp", "")
                    sender = packet.get("sender", "unknown")
                    errornum = packet.get("errornum", 0)

                    log_msg = f"[{sender}] [{timestamp}] (errornum={errornum}) {message}"
                    self.logger.error(log_msg)

                    # TODO: no error-handling policy decided yet (retry,
                    # alert, restart the offending worker, ...) - logging
                    # is the only action taken for now.

                    self.error_queue.task_done()
            except queue.Empty:
                continue
        print("Error handler thread shutting down...")

    # --- Heartbeat Thread ---
    def heartbeat_thread(self):
        print("Heartbeat thread started.")
        while not self.shutdown_event.is_set():
            try:
                packet = self.heartbeat_queue.get(timeout=1.0)
                if packet is not None:
                    worker_name = packet.get("sender", "unknown_worker")
                    timestamp = packet.get("timestamp", None)

                    self.heartbeat_tracker.record_heartbeat(worker_name, timestamp)

                    # Publish aggregated heartbeat map to Redis so the Web API can serve it
                    try:
                        self.r.set("mtbot:heartbeats", json.dumps(self.heartbeat_tracker.get_status()))
                    except Exception:
                        pass

                    self.heartbeat_queue.task_done()
            except queue.Empty:
                continue
        print("Heartbeat thread shutting down...")
    
    def register_signals(self):
        import signal
        def handle_signal(signum, frame):
            if signum in (signal.SIGINT, signal.SIGTERM):
                print(f"Manager received termination signal {signum}. Initiating shutdown sequence...")
                self.shutdown_sequence()
            elif hasattr(signal, "SIGUSR1") and signum == signal.SIGUSR1:
                print("Manager received SIGUSR1. Broadcasting STOP command...")
                self.broadcast_command("STOP")
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, handle_signal)
    
    # --- Bootstrapper ---
    def start(self):
        print("Starting Manager...")
        
        # Register signals
        self.register_signals()
        
        # Start helper threads
        threading.Thread(target=self.logger_thread, daemon=True).start()
        threading.Thread(target=self.heartbeat_thread, daemon=True).start()
        threading.Thread(target=self.error_handler_thread, daemon=True).start()

        # Start Redis listener thread
        threading.Thread(target=self.redis_manager.listen, args=("manager_queue", 1), daemon=True).start()

        # Instantiate workers passing SHM details where required
        self.datapipeline = DataPipeline("DataPipeline", shm_name=self.shm_name, shm_lock=self.shm_lock)
        self.analyser = Analyser("Analyser", shm_name=self.shm_name, shm_lock=self.shm_lock)
        self.executer = Executer("Executer", shm_name=self.shm_name, shm_lock=self.shm_lock)
        
        self.children = [self.datapipeline, self.analyser, self.executer]

        for child in self.children:
            child.start()

        try:
            for child in self.children:
                child.join()
        except KeyboardInterrupt:
            self.shutdown_sequence()

    def shutdown_sequence(self):
        if getattr(self, '_shutting_down', False):
            return
        self._shutting_down = True
        print("Initiating system shutdown sequence...")
        
        # Set shutdown event to stop logger and heartbeat threads
        self.shutdown_event.set()
        
        # 1. Kill Analyser immediately
        if hasattr(self, 'analyser') and self.analyser.is_alive():
            print("Terminating Analyser process...")
            try:
                self.analyser.terminate()
                self.analyser.join()
                print("Analyser process terminated.")
            except Exception as e:
                print(f"Error terminating Analyser: {e}")
            
        # 2. Send SELL_ALL command to Executer via UDS
        executer_socket_path = "/tmp/executer.sock"
        if hasattr(self, 'executer') and self.executer.is_alive() and os.path.exists(executer_socket_path):
            print("Sending SELL_ALL command to Executer...")
            try:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(executer_socket_path)
                packet = MPacket.create("Manager", "Executer", "SELL_ALL")
                client.sendall((packet.to_json() + "\n").encode('utf-8'))
                client.close()
                print("SELL_ALL command sent. Waiting for Executer to liquidate and shutdown...")
                self.executer.join(timeout=5)
            except Exception as e:
                print(f"Failed to communicate with Executer during shutdown: {e}")
                
        # Make sure Executer is stopped
        if hasattr(self, 'executer') and self.executer.is_alive():
            print("Force terminating Executer process...")
            try:
                self.executer.terminate()
                self.executer.join()
            except Exception as e:
                print(f"Error terminating Executer: {e}")
            
        # 3. Terminate DataPipeline
        if hasattr(self, 'datapipeline') and self.datapipeline.is_alive():
            print("Terminating DataPipeline process...")
            try:
                self.datapipeline.terminate()
                self.datapipeline.join()
            except Exception as e:
                print(f"Error terminating DataPipeline: {e}")
            
        # 4. Clean up RedisManager
        self.redis_manager.stop_listening()
        
        # 5. Clean up shared memory using shm_service
        try:
            shm_service.cleanup()
            print("Shared portfolio memory unlinked successfully.")
        except Exception as e:
            print(f"Error unlinking shared memory: {e}")
            
        # Clean up sockets if they exist
        for path in ["/tmp/analyser.sock", "/tmp/executer.sock", "/tmp/data_pipeline_service.sock"]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
        print("System shutdown complete.")

    def broadcast_command(self, action, payload=None):
        packet = MPacket.create("Manager", "ALL", action, payload=payload)
        try:
            self.r.publish("broadcast_channel", packet.to_json())
            print(f"[BROADCAST] Sent action: {action}")
        except Exception as e:
            print(f"[ERROR] Failed to publish broadcast command: {e}")
