"""
Whole-system communication test: does a message a worker sends actually
arrive at Manager, correctly attributed to that worker, using the new
MPacket `sender`/`to` schema? And does a command Manager broadcasts
actually reach a worker and get dispatched to the right handler?

This is intentionally scoped to what's true TODAY, not the end state. The
inter-process protocol unification (see the plan at
~/.claude/plans/jaunty-sparking-wilkes.md, since refined in conversation)
is being done in stages - MPacket schema, BaseWorker's register_handler/
dispatch wired into check_broadcast, and Manager's error/log split are done
so far. Still to come: Executer's order channel and the DataPipeline->
Analyser dollar_bar channel onto the same envelope+dispatch pattern. As
each lands, add a test here covering it, so this file ends up being the
"does the whole system actually talk to itself correctly, end to end" test.

Right now this proves:
- a worker's send_log()/send_heartbeat() travel over real Redis, through
  RedisManager.listen()'s routing, into a REAL running Manager, correctly
  attributed via `sender`.
- a plain log (errornum == 0) and an error-flagged one (errornum > 0) take
  different paths inside Manager - log_queue/logger_thread vs.
  error_queue/error_handler_thread - and both still end up logged.
- Manager.broadcast_command() reaches a real worker's check_broadcast()
  over real Redis pub/sub, and dispatch() correctly routes it to
  BaseWorker's default STOP handler.
"""
import logging
import threading
import time

import pytest

from manager.manager import Manager
from workers.base_worker import BaseWorker

from _helpers import wait_until, make_bare_worker


class _CapturingLogHandler(logging.Handler):
    """
    Collects formatted log records in memory instead of relying on
    Manager's real log file (manager/logger_c.py writes to logs/*.log on
    disk) - lets the test assert on what Manager actually logged without
    parsing a file.
    """
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


@pytest.fixture
def running_manager(monkeypatch, redis_test_port, isolated_shm, no_binance_keys):
    """
    Boot a REAL Manager instance, pointed at the disposable test Redis, with
    its background threads running - but without going through
    Manager.start()/register_signals() (signal.signal() only works on the
    process's main thread; we start Manager's worker threads directly here
    instead, the same way _helpers.start_worker_in_background_thread does
    for BaseWorker subclasses).

    Manager reads REDIS_HOST/REDIS_PORT from the environment directly in
    __init__ (unlike BaseWorker, which hardcodes them as class constants -
    see the note in conftest.py's patched_worker_redis_target fixture), so
    isolating it just means setting those env vars before constructing it.
    """
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", str(redis_test_port))

    manager = Manager()

    capture = _CapturingLogHandler()
    logging.getLogger("QuantSystem").addHandler(capture)

    threads = [
        threading.Thread(target=manager.logger_thread, daemon=True),
        threading.Thread(target=manager.heartbeat_thread, daemon=True),
        threading.Thread(target=manager.error_handler_thread, daemon=True),
        threading.Thread(target=manager.redis_manager.listen, args=("manager_queue", 1), daemon=True),
    ]
    for t in threads:
        t.start()

    yield manager, capture

    manager.shutdown_event.set()
    manager.redis_manager.stop_listening()
    # redis_manager.listen() blocks inside blpop(timeout=1) between checks
    # of `running`, so without joining here, this Manager's threads can
    # still be alive into the NEXT test - and since manager_queue is the
    # same disposable test Redis for the whole session, a still-running
    # logger_thread can steal a packet another test just pushed before
    # that test gets to assert on it. Join with headroom past the 1s
    # blpop timeout so teardown only returns once everything has actually
    # stopped.
    for t in threads:
        t.join(timeout=2)
    logging.getLogger("QuantSystem").removeHandler(capture)


def test_worker_heartbeat_is_correctly_attributed_by_manager(
    redis_client, running_manager, redis_test_port,
):
    manager, _ = running_manager
    worker = make_bare_worker("TestWorkerX", redis_test_port)

    worker.send_heartbeat()

    wait_until(
        lambda: "TestWorkerX" in manager.heartbeat_tracker.get_status(),
        message="Manager's heartbeat tracker to record TestWorkerX",
    )


def test_worker_info_log_is_correctly_attributed_by_manager(
    redis_client, running_manager, redis_test_port,
):
    """A plain (errornum == 0) log travels through log_queue/logger_thread."""
    manager, capture = running_manager
    worker = make_bare_worker("TestWorkerY", redis_test_port)

    worker.send_log("started up fine")

    wait_until(
        lambda: any("TestWorkerY" in line and "started up fine" in line for line in capture.records),
        message="Manager's logger to record a correctly-attributed message from TestWorkerY",
    )


def test_worker_error_is_routed_to_error_handler_thread(
    redis_client, running_manager, redis_test_port,
):
    """
    An error-flagged (errornum > 0) log takes a different path than a plain
    one: RedisManager.listen() routes it to error_queue instead of
    log_queue, and Manager.error_handler_thread (not logger_thread) is what
    logs it - currently the only "action" that thread takes, since no real
    error-handling policy has been decided yet.
    """
    manager, capture = running_manager
    worker = make_bare_worker("TestWorkerY", redis_test_port)

    worker.send_log("something went sideways", errnum=1)

    wait_until(
        lambda: any(
            "TestWorkerY" in line and "something went sideways" in line and "errornum=1" in line
            for line in capture.records
        ),
        message="Manager's error handler to record a correctly-attributed error from TestWorkerY",
    )


def test_manager_broadcast_reaches_a_real_worker_and_dispatches(
    redis_client, running_manager, redis_test_port,
):
    """
    End-to-end: Manager.broadcast_command("STOP") -> Redis pub/sub
    broadcast_channel -> a real worker's check_broadcast() decodes the
    MPacket -> dispatch() routes it to BaseWorker's default STOP handler.
    This is the first channel fully migrated off the old ad-hoc
    {"action": ...} dict (see manager/manager.py broadcast_command and
    workers/base_worker.py check_broadcast).
    """
    manager, _ = running_manager

    worker = make_bare_worker("TestWorkerZ", redis_test_port)
    worker.pubsub = worker.r.pubsub(ignore_subscribe_messages=True)
    worker.pubsub.subscribe(BaseWorker.BROADCAST_CHANNEL)
    worker.shutdown = False
    worker.stop_buying = False

    # redis-py's subscribe() returns once the SUBSCRIBE command is sent,
    # not once Redis has necessarily finished registering it against this
    # connection before a PUBLISH from a different connection arrives -
    # pub/sub has no backlog/replay for late subscribers, so a message
    # published in that gap would be silently lost. A short settle delay
    # is the standard, pragmatic way to avoid that race in a test.
    time.sleep(0.2)

    manager.broadcast_command("STOP")

    def worker_received_stop():
        worker.check_broadcast()
        return worker.stop_buying is True

    wait_until(worker_received_stop, message="worker to receive and dispatch a broadcast STOP command")
