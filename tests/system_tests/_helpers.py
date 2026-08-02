"""
Small shared utilities for the system-level integration tests in this folder.

This is NOT a test module itself (its filename doesn't start with `test_`),
so pytest won't try to collect it - it's just a helper library that the real
test_*.py files in this directory import from.
"""
import json
import socket
import threading
import time

import redis

from core.packets.m_packet import MPacket
from workers.base_worker import BaseWorker


def find_free_port() -> int:
    """Ask the OS for an unused TCP port, so parallel test runs never collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.05, message: str = "condition"):
    """
    Poll `predicate()` until it returns something truthy, or raise after `timeout` seconds.

    Integration tests like these are inherently timing-based - we're waiting for
    another thread/process to do something over a socket, Redis, or shared
    memory. Polling frequently and failing with a clear message beats a blind
    `time.sleep(n)` guess: it's both faster (returns as soon as the condition
    is true) and more honest about what it was actually waiting for.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    raise AssertionError(f"Timed out after {timeout}s waiting for: {message}")


def start_worker_in_background_thread(worker: BaseWorker, redis_port: int) -> threading.Thread:
    """
    Run a BaseWorker subclass's `work()` loop on a background thread inside
    THIS test process, instead of spawning it as a real OS subprocess via
    `.start()`.

    Why not just use the real `.start()` / `.run()` path?
    - `BaseWorker.run()` calls `signal.signal(...)`, and signal handlers can
      only be registered on the main thread of the main process - calling
      `.run()` from a background thread raises immediately.
    - Running in-process (as a thread) lets tests `monkeypatch` things like
      `BinanceClientService` directly on the classes the worker uses, which
      only works because the worker shares our interpreter and our patched
      state. A real subprocess re-imports everything fresh via fork/spawn
      and would need the patches applied *before* the fork instead.

    We replicate the (non-signal) parts of BaseWorker.run() by hand here.
    """
    worker.r = redis.Redis(host="127.0.0.1", port=redis_port)
    worker.pubsub = worker.r.pubsub(ignore_subscribe_messages=True)
    worker.pubsub.subscribe(BaseWorker.BROADCAST_CHANNEL)
    worker.shutdown = False
    worker.stop_buying = False

    thread = threading.Thread(target=worker.work, daemon=True)
    thread.start()
    return thread


def make_bare_worker(worker_name: str, redis_port: int) -> BaseWorker:
    """
    A bare BaseWorker instance with just enough set up to call
    send_log()/send_heartbeat()/dispatch() directly - those only touch
    `self.r`, so we don't need the full work()-loop setup
    start_worker_in_background_thread() does for tests that aren't
    exercising a real pub/sub subscription or UDS socket.
    """
    worker = BaseWorker(worker_name)
    worker.r = redis.Redis(host="127.0.0.1", port=redis_port)
    return worker


def send_uds_line(socket_path: str, payload: dict) -> None:
    """
    Send one fire-and-forget JSON line to a Unix Domain Socket server and
    close the connection - the exact same connect/send/close pattern
    production code uses in ExecutorService.send_order().
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    sock.close()


def send_mpacket_uds_line(socket_path: str, sender: str, to: str, msg_type: str, payload: dict = None) -> None:
    """
    Build a real MPacket and send it as one fire-and-forget UDS line -
    what Analyser/Manager actually send to Executer now that its order
    channel is on the MPacket + dispatch protocol (raw {"action": ...}
    dicts are no longer recognized).
    """
    packet = MPacket.create(sender, to, msg_type, payload=payload)
    send_uds_line(socket_path, packet.to_dict())


def send_raw_uds_bytes(socket_path: str, raw_bytes: bytes) -> None:
    """
    Send raw bytes (not JSON-encoded) to a UDS server and close the
    connection - for exercising the layer BEFORE JSON/MPacket parsing,
    e.g. invalid UTF-8 that breaks `data.decode('utf-8')` itself.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    sock.sendall(raw_bytes)
    sock.close()


def drain_manager_queue(redis_client, max_items: int = 1000) -> list:
    """Pop every packet currently sitting in the `manager_queue` Redis list."""
    packets = []
    for _ in range(max_items):
        raw = redis_client.lpop("manager_queue")
        if raw is None:
            break
        packets.append(json.loads(raw))
    return packets
