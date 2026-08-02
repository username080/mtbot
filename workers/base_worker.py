import redis
import multiprocessing
from core.shm import shm_service
from core.packets.m_packet import MPacket
from core.packets.dispatcher import MessageDispatcherMixin
from core.exceptions.custom_exceptions import SHMError
from core.exceptions.exception_handler import handle_exception

class BaseWorker(MessageDispatcherMixin, multiprocessing.Process):
    # Redis configuration
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    BROADCAST_CHANNEL = "broadcast_channel"
    
    # Shared memory configuration
    SHM_SIZE = 1024

    def __init__(self, name, shm_name=None, shm_lock=None):
        super().__init__()
        self.worker_name = name
        self.shm_name = shm_name
        self.shm_lock = shm_lock
        self.heartbeat_period = 2
        
        # Configure shm_service wrapper with parameters passed from Manager
        if shm_name is not None:
            shm_service.shm_name = shm_name
        if shm_lock is not None:
            shm_service.lock = shm_lock

        # Message dispatch: subclasses register additional handlers (in
        # __init__ or work()) for whatever message types they care about
        # beyond these three, which are common to every worker.
        self._handlers = {}
        self.register_handler("SHUTDOWN", self._handle_shutdown)
        self.register_handler("RESUME", self._handle_resume)
        self.register_handler("STOP", self._handle_stop)

    def _handle_shutdown(self, packet: MPacket) -> None:
        self.shutdown = True

    def _handle_resume(self, packet: MPacket) -> None:
        self.shutdown = False
        self.stop_buying = False

    def _handle_stop(self, packet: MPacket) -> None:
        self.stop_buying = True

    def run(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)

        self.r = redis.Redis(host=self.REDIS_HOST, port=self.REDIS_PORT, db=self.REDIS_DB)
        self.pubsub = self.r.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(self.BROADCAST_CHANNEL)
        self.shutdown = False
        self.stop_buying = False
        self.send_log(f"{self.worker_name} initialized and running.")
        try:
            self.work()
        except Exception as e:
            handle_exception(e, context_worker=self)

    def check_broadcast(self):
        try:
            msg = self.pubsub.get_message()
            if msg:
                packet = MPacket.from_json(msg['data'].decode('utf-8'))
                self.send_log(f"Received broadcast command: {packet.type}")
                self.dispatch(packet)
        except Exception as e:
            self.send_log(f"Error checking pubsub: {e}", errnum=1)

    def send_log(self, message, errnum=0):
        packet = MPacket.create(self.worker_name, "Manager", "log", errornum=errnum, msg=message)
        self.r.lpush("manager_queue", packet.to_json())

    def send_heartbeat(self):
        packet = MPacket.create(self.worker_name, "Manager", "heartbeat", msg=self.worker_name)
        self.r.lpush("manager_queue", packet.to_json())

    def work(self):
        pass