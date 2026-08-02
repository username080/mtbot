"""
Regression tests for the Executer's blocking, single-threaded main loop.

Executer.work() (workers/executer/executer.py) runs one big `while` loop
that does ALL of the following, one step after another, on a single thread:
  - check broadcast commands (STOP / RESUME / SHUTDOWN)
  - send heartbeats
  - accept new UDS connections
  - read and process every pending order from every connected client

The problem: processing a single BUY/SELL/SMART_EXIT order can call
`_execute_soft_chase()` or `_execute_smart_exit()`, and those methods
contain their own `while ...: time.sleep(...)` loops that can run for tens
of seconds (up to ~60s for a chase, ~110s worst case for a smart exit).
Because everything shares one thread, NOTHING else happens while that's in
flight - not heartbeats, not STOP/SHUTDOWN broadcasts, not a second
symbol's order.

These tests replace `_execute_soft_chase` with a stand-in that blocks the
calling thread for a few real seconds before returning `True` - the same
"runs synchronously for a while, then returns" *shape* as the real method,
without needing to fake a full sequence of Binance REST responses or fight
with the real method's internal 60-second timers. What's under test here is
NOT the soft-chase pricing/pegging logic (that deserves its own unit tests)
- it's whether Executer.work()'s single thread can do anything else while
ANY synchronous order-execution call is in flight. The real method and this
stand-in share the only property these tests rely on.

Both tests describe the behavior we WANT (heartbeats and a second symbol's
order keep flowing even while one order is being executed) and are marked
`xfail(strict=True)`, because that is not what happens today. Once
Executer's order execution is moved off the main loop (e.g. a thread or
asyncio task per order), these tests should start passing - pytest will
then report them as "XPASS" (an xfail that unexpectedly passed), and
`strict=True` turns that into a hard failure. That failure is your cue to
come back here and delete the `xfail` marker: it means the fix worked and
this is now a real, permanently-enforced regression test.
"""
import os
import time

import pytest

from workers.executer.executer import Executer

from _helpers import wait_until, start_worker_in_background_thread, send_mpacket_uds_line, drain_manager_queue

# How long the stubbed-out BTCUSDT execution blocks the Executer's main
# thread for, in real seconds. Short enough to keep the test suite fast,
# long enough to be unmistakably longer than a couple of heartbeat periods.
BLOCK_SECONDS = 3.0


@pytest.fixture
def stub_slow_btc_execution(monkeypatch):
    """
    Replace `_execute_soft_chase` so that BTCUSDT orders block for
    BLOCK_SECONDS before "filling", while any other symbol (e.g. ETHUSDT)
    fills immediately. This lets the two tests below use BTCUSDT as the
    "slow order in flight" and ETHUSDT as the "should this get starved?"
    probe, without needing real Binance responses.
    """
    calls = []

    def fake_soft_chase(self, symbol, side, quantity, leverage, signal_price, sl_price, tp_price):
        calls.append(symbol)
        if symbol == "BTCUSDT":
            time.sleep(BLOCK_SECONDS)
        return True

    monkeypatch.setattr(Executer, "_execute_soft_chase", fake_soft_chase)
    return calls


@pytest.fixture
def running_executer(tmp_path, patched_worker_redis_target, isolated_shm, no_binance_keys):
    """
    Boot a real Executer instance on a background thread (see
    _helpers.start_worker_in_background_thread for why a thread, not a
    subprocess), wired to an isolated UDS socket path, isolated shared
    memory, and the disposable test Redis - then tear it down cleanly.
    """
    socket_path = str(tmp_path / "executer_test.sock")

    worker = Executer("TestExecuter")
    worker.UDS_SOCKET_PATH = socket_path
    worker.HEARTBEAT_PERIOD = 0.4  # fast heartbeat so a 3s stall is easy to detect precisely

    thread = start_worker_in_background_thread(worker, patched_worker_redis_target)
    wait_until(lambda: os.path.exists(socket_path), message="Executer UDS socket to appear")

    yield worker, socket_path

    worker.stopped = True
    thread.join(timeout=2)


@pytest.mark.xfail(
    strict=True,
    reason="Executer.work() executes orders inline on its one thread, so a "
           "single slow order blocks heartbeats for its entire duration.",
)
def test_heartbeat_keeps_flowing_during_a_slow_order(
    redis_client, running_executer, stub_slow_btc_execution,
):
    worker, socket_path = running_executer

    # Confirm the loop is alive and already heartbeating before we do anything
    # that might block it - this part should always succeed.
    wait_until(
        lambda: any(p.get("type") == "heartbeat" for p in drain_manager_queue(redis_client)),
        message="first heartbeat",
    )

    # Trigger the ~BLOCK_SECONDS-long stubbed BTCUSDT execution.
    send_mpacket_uds_line(socket_path, "Analyser", "Executer", "BUY", {
        "symbol": "BTCUSDT", "quantity": 0.01, "price": 99900.0,
    })

    # While that's in flight, track the largest gap between consecutive
    # heartbeats we observe.
    last_heartbeat_at = time.time()
    max_gap = 0.0
    observe_until = time.time() + BLOCK_SECONDS + 1.5
    while time.time() < observe_until:
        now = time.time()
        for packet in drain_manager_queue(redis_client):
            if packet.get("type") == "heartbeat":
                max_gap = max(max_gap, now - last_heartbeat_at)
                last_heartbeat_at = now
        time.sleep(0.1)

    # Desired behavior: heartbeats should never fall more than ~3 periods
    # behind, even while an order is executing. Today, max_gap ends up
    # roughly equal to BLOCK_SECONDS, because nothing else runs until the
    # stubbed execution call returns.
    assert max_gap < worker.HEARTBEAT_PERIOD * 3, (
        f"Heartbeat gap of {max_gap:.2f}s observed - the main loop appears to "
        f"have stalled while an order was executing."
    )


@pytest.mark.xfail(
    strict=True,
    reason="Executer.work() only accept()s/reads one connection's data at a "
           "time in a single loop iteration; a second symbol's order sits "
           "unread until the first symbol's synchronous execution returns.",
)
def test_second_symbols_order_is_not_blocked_by_first_symbols_execution(
    redis_client, isolated_shm, running_executer, stub_slow_btc_execution,
):
    worker, socket_path = running_executer

    # Trigger the slow BTCUSDT order first ...
    send_mpacket_uds_line(socket_path, "Analyser", "Executer", "BUY", {
        "symbol": "BTCUSDT", "quantity": 0.01, "price": 99900.0,
    })
    time.sleep(0.3)  # give the Executer a moment to accept + start executing it

    # ... then immediately send a completely independent ETHUSDT order.
    eth_sent_at = time.time()
    send_mpacket_uds_line(socket_path, "Analyser", "Executer", "BUY", {
        "symbol": "ETHUSDT", "quantity": 0.1, "price": 99900.0,
    })

    def eth_position_exists():
        portfolio = isolated_shm.read_portfolio()
        return abs(portfolio.get("positions", {}).get("ETHUSDT", 0.0)) > 0

    wait_until(eth_position_exists, timeout=BLOCK_SECONDS + 2, message="ETHUSDT position to appear")
    eth_processed_after = time.time() - eth_sent_at

    # Desired behavior: ETHUSDT's order is independent of BTCUSDT's and
    # should be picked up almost immediately (well under one second), not
    # after BTCUSDT's entire ~BLOCK_SECONDS execution has elapsed.
    assert eth_processed_after < 1.0, (
        f"ETHUSDT order took {eth_processed_after:.2f}s to be processed - it "
        f"looks like it had to wait for BTCUSDT's execution to finish first."
    )
