"""
Regression tests for what happens when a worker process hits a bad input.

Three things are checked here:

1. That a malformed order field (a non-numeric "price") no longer crashes
   Executer. This used to kill the whole process: `float("not-a-number")`
   raises `ValueError`, and the old inline if/elif in Executer.work()'s
   connection-read loop only guarded against `json.JSONDecodeError` - the
   ValueError propagated all the way up to BaseWorker.run()'s
   `except Exception: handle_exception(...)`, which just logs and lets the
   process die. Now that order handling lives in a registered handler
   (Executer._handle_order) invoked through BaseWorker.dispatch(), which
   wraps every handler call in try/except and logs failures via
   send_log(errnum=1, ...) instead of letting them escape, the same bad
   input just gets logged - the process survives. This test asserts that
   fix directly.

2. That a DIFFERENT, still-real crash vector exists: dispatch()'s
   try/except only wraps handler invocation - it can't help with something
   that fails before a packet even exists. Executer.work()'s
   `data.decode('utf-8')` has no guard around it at all, so bytes that
   aren't valid UTF-8 raise UnicodeDecodeError right there (a ValueError
   subclass, not caught by the surrounding `except BlockingIOError`/
   `except socket.error`), which still kills the process today.

3. That nothing brings the process back afterwards (using that same
   invalid-UTF-8 crash). Manager.start() (manager/manager.py) only ever
   calls `child.start()` once and then `child.join()` - there is no
   supervisor loop watching for a dead child and restarting it. This part
   is marked `xfail(strict=True)`: it describes the behavior we eventually
   want (automatic recovery), which does not exist yet. Once a supervisor
   is added to Manager, this assertion should start passing - pytest will
   report an unexpected XPASS, and `strict=True` turns that into a
   failure. That's your cue to delete the xfail marker: the fix works, and
   this becomes a real, permanently-enforced regression test.

Unlike the other test files in this folder, this one spins up a REAL OS
subprocess (multiprocessing.Process, via Executer.start()) rather than a
background thread, because "did the process actually die" is inherently a
process-level fact, not something a thread can demonstrate.
"""
import multiprocessing
import os

import pytest

from workers.executer.executer import Executer

from _helpers import wait_until, send_mpacket_uds_line, send_raw_uds_bytes, drain_manager_queue


@pytest.fixture(autouse=True)
def _require_fork_start_method():
    """
    This file relies on Linux's default 'fork' multiprocessing start
    method: fixtures monkeypatch class attributes (BaseWorker.REDIS_HOST/
    PORT) and a module-level singleton (shm_service's segment names) in
    THIS process *before* `Executer.start()` is called, and rely on fork()
    copying that already-patched state into the child process. A 'spawn'
    start method would instead re-import everything fresh in the child and
    silently lose our patches, so we skip rather than give a false result.
    """
    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("This test requires the 'fork' multiprocessing start method (Linux default).")


def test_malformed_order_field_does_not_crash_the_executer_process(
    tmp_path, patched_worker_redis_target, isolated_shm, no_binance_keys, redis_client,
):
    socket_path = str(tmp_path / "executer_crash_test.sock")

    worker = Executer("CrashTestExecuter")
    worker.UDS_SOCKET_PATH = socket_path
    worker.start()  # real OS subprocess, via fork

    try:
        wait_until(lambda: os.path.exists(socket_path), message="Executer UDS socket to appear")

        # "price" should be numeric. float("not-a-number") raises
        # ValueError inside Executer._handle_order - dispatch() now
        # catches that.
        send_mpacket_uds_line(socket_path, "Analyser", "Executer", "BUY", {
            "symbol": "BTCUSDT", "quantity": 1, "price": "not-a-number",
        })

        error_packets = wait_until(
            lambda: [p for p in drain_manager_queue(redis_client) if p.get("errornum", 0) > 0] or None,
            timeout=5,
            message="a dispatch-caught error log packet on manager_queue",
        )
        assert any("ValueError" in p.get("msg", "") for p in error_packets), (
            f"Expected dispatch() to have logged a ValueError, got: {error_packets}"
        )

        # The actual point of this test: the process is still alive.
        assert worker.is_alive(), (
            "Executer process died even though dispatch() should have caught the handler's ValueError."
        )
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)


def test_invalid_utf8_bytes_still_crash_the_executer_process(
    tmp_path, patched_worker_redis_target, isolated_shm, no_binance_keys,
):
    socket_path = str(tmp_path / "executer_crash_test.sock")

    worker = Executer("CrashTestExecuter")
    worker.UDS_SOCKET_PATH = socket_path
    worker.start()

    try:
        wait_until(lambda: os.path.exists(socket_path), message="Executer UDS socket to appear")

        # 0xFF is never a valid UTF-8 leading byte - data.decode('utf-8')
        # raises UnicodeDecodeError before any JSON/MPacket parsing (and
        # therefore before dispatch()'s try/except) is ever reached.
        send_raw_uds_bytes(socket_path, b"\xff\xfe\xfd not valid utf-8\n")

        wait_until(lambda: not worker.is_alive(), timeout=5, message="Executer process to die")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)


@pytest.mark.xfail(
    strict=True,
    reason="Manager only calls child.join() on its workers after starting "
           "them once - there is no supervisor that detects a dead worker "
           "and restarts it.",
)
def test_executer_is_restarted_after_crashing(
    tmp_path, patched_worker_redis_target, isolated_shm, no_binance_keys,
):
    socket_path = str(tmp_path / "executer_crash_test.sock")

    worker = Executer("CrashTestExecuter")
    worker.UDS_SOCKET_PATH = socket_path
    worker.start()

    try:
        wait_until(lambda: os.path.exists(socket_path), message="Executer UDS socket to appear")

        send_raw_uds_bytes(socket_path, b"\xff\xfe\xfd not valid utf-8\n")
        wait_until(lambda: not worker.is_alive(), timeout=5, message="Executer process to die")

        # `os.unlink`s the socket file in its `finally` block when it exits, so
        # the socket path only exists again if *something* rebuilds an
        # Executer (or equivalent) and rebinds it. Desired behavior: a
        # supervisor notices the crash and brings a replacement up within a
        # few heartbeat periods.
        wait_until(
            lambda: os.path.exists(socket_path),
            timeout=6,
            message="a replacement Executer socket to appear after the crash",
        )
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
