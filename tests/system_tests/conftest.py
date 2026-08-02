"""
Shared fixtures for the system-level integration tests.

These tests deliberately use REAL Redis, REAL Unix domain sockets, and REAL
POSIX shared memory instead of mocking them away, because the bugs under
test (dead wiring, blocking I/O starving the control plane, missing crash
supervision) only show up in how the real pieces interact with each other -
a fully-mocked unit test would hide exactly what we're trying to catch.

To make that safe to run on a dev machine - and safe to run right next to a
*real*, already-running instance of the bot - every fixture below is careful
to use throwaway ports, throwaway socket paths, and uniquely-named shared
memory segments, instead of the hardcoded values (localhost:6379,
/tmp/executer.sock, "portfolio_shm", ...) the production code defaults to.
"""
import subprocess
import time
import uuid

import pytest
import redis

from core.shm import shm_service
from workers.base_worker import BaseWorker

from _helpers import find_free_port


@pytest.fixture(scope="session")
def redis_test_port():
    """
    Start one disposable redis-server for the whole test session (real
    Redis, not a fake/in-memory stand-in - the production RedisManager
    relies on real blocking-list and pub/sub semantics we want to exercise
    faithfully).
    """
    port = find_free_port()
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    client = redis.Redis(host="127.0.0.1", port=port)
    deadline = time.time() + 5
    up = False
    while time.time() < deadline:
        try:
            if client.ping():
                up = True
                break
        except redis.ConnectionError:
            time.sleep(0.1)

    if not up:
        proc.terminate()
        pytest.fail("Test redis-server did not come up in time - is redis-server installed?")

    yield port

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def redis_client(redis_test_port):
    """A client pointed at the disposable test Redis, flushed clean before use."""
    client = redis.Redis(host="127.0.0.1", port=redis_test_port)
    client.flushall()
    yield client
    client.flushall()


@pytest.fixture
def patched_worker_redis_target(monkeypatch, redis_test_port):
    """
    Point BaseWorker (and therefore every worker subclass: DataPipeline,
    Analyser, Executer) at the disposable test Redis instead of
    localhost:6379.

    Side note for later: BaseWorker.REDIS_HOST / REDIS_PORT are hardcoded
    class constants, not read from the REDIS_HOST / REDIS_PORT environment
    variables that Manager itself respects (manager/manager.py reads
    os.getenv("REDIS_HOST", ...) for its own client). That's a real
    integration bug we have to work around here via monkeypatch: in a
    Docker deployment where REDIS_HOST != localhost, every worker process
    would currently still try to reach localhost:6379 and fail to connect,
    even though the Manager itself connects to the right host.
    """
    monkeypatch.setattr(BaseWorker, "REDIS_HOST", "127.0.0.1")
    monkeypatch.setattr(BaseWorker, "REDIS_PORT", redis_test_port)
    return redis_test_port


@pytest.fixture
def isolated_shm(monkeypatch):
    """
    Give this test its own private set of shared-memory segments instead of
    the real "portfolio_shm" / "positions_info_shm" names, so tests can't
    collide with each other (pytest-xdist / repeated runs) or with a real
    bot instance running on the same machine.
    """
    suffix = uuid.uuid4().hex[:8]
    monkeypatch.setattr(shm_service, "portfolio_name", f"test_portfolio_{suffix}")
    monkeypatch.setattr(shm_service, "positions_info_name", f"test_positions_{suffix}")

    shm_service.init_shm({
        "portfolio": {"cash": 100_000.0, "positions": {}},
        "positions_info": {},
    })
    yield shm_service
    shm_service.cleanup()


@pytest.fixture
def no_binance_keys(monkeypatch):
    """Make sure these tests never accidentally hit the real Binance API."""
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
