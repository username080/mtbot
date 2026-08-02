"""
Unit tests for the generic message dispatch mechanism added to BaseWorker
(core/packets/dispatcher.py + BaseWorker.register_handler/dispatch).

These don't need a running worker process/thread, real UDS sockets, or
shared memory - dispatch() only touches self._handlers and self.send_log(),
so a bare BaseWorker instance with just a Redis client attached
(_helpers.make_bare_worker) is enough to exercise it for real: send_log()
really does push to Redis, and these tests assert on what actually landed
in manager_queue rather than mocking it away.
"""
import json

from core.packets.m_packet import MPacket

from _helpers import make_bare_worker


def test_dispatch_calls_the_registered_handler_with_the_packet(redis_client, redis_test_port):
    worker = make_bare_worker("TestDispatch", redis_test_port)
    received = []
    worker.register_handler("PING", lambda packet: received.append(packet))

    packet = MPacket.create("Analyser", "TestDispatch", "PING", payload={"n": 1})
    worker.dispatch(packet)

    assert len(received) == 1
    assert received[0] is packet


def test_unregistered_message_type_does_not_raise_and_gets_logged(redis_client, redis_test_port):
    worker = make_bare_worker("TestDispatch", redis_test_port)

    packet = MPacket.create("Analyser", "TestDispatch", "SOMETHING_NOBODY_HANDLES")
    worker.dispatch(packet)  # must not raise

    logged = redis_client.lrange("manager_queue", 0, -1)
    assert any(b"No handler registered" in item for item in logged)


def test_handler_exception_is_caught_and_logged_not_propagated(redis_client, redis_test_port):
    """
    This is the property that fixes the crash bug documented in
    test_crash_supervision.py: a handler raising must never be able to
    kill the worker process that's dispatching to it. Once Executer's
    order handling is migrated onto register_handler/dispatch (step 4),
    that test's "process dies" assertion should flip to "process survives"
    thanks to this.
    """
    worker = make_bare_worker("TestDispatch", redis_test_port)

    def bad_handler(packet):
        raise ValueError("boom")

    worker.register_handler("BOOM", bad_handler)
    packet = MPacket.create("Analyser", "TestDispatch", "BOOM")

    worker.dispatch(packet)  # must not raise

    logged = [json.loads(item) for item in redis_client.lrange("manager_queue", 0, -1)]
    assert any(p["errornum"] > 0 and "ValueError" in p["msg"] for p in logged)


def test_default_shutdown_resume_stop_handlers_flip_the_right_flags(redis_client, redis_test_port):
    worker = make_bare_worker("TestDispatch", redis_test_port)
    worker.shutdown = False
    worker.stop_buying = False

    worker.dispatch(MPacket.create("Manager", "ALL", "STOP"))
    assert worker.stop_buying is True

    worker.dispatch(MPacket.create("Manager", "ALL", "SHUTDOWN"))
    assert worker.shutdown is True

    worker.dispatch(MPacket.create("Manager", "ALL", "RESUME"))
    assert worker.shutdown is False
    assert worker.stop_buying is False
