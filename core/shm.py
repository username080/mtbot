import json
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from core.exceptions.custom_exceptions import SHMError

class SharedMemoryService:
    """
    Service class to wrap shared memory read/write logic for two separate
    segments (portfolio, positions_info), each guarded by its own
    multiprocessing lock to eliminate contentions.

    There used to be a third segment, tracked_coins, written by Executer
    and DataPipeline - but nothing ever read `state` except DataPipeline's
    own shm_check_loop (dynamic unsubscribe on trade close, which doesn't
    make sense for a fixed BTC/ETH symbol universe), and nothing ever read
    `ongoing_trade` back to make a decision at all. It was removed rather
    than kept as write-only bookkeeping nobody consults.
    """
    def __init__(self):
        # 1. Segment names
        self.portfolio_name = "portfolio_shm"
        self.positions_info_name = "positions_info_shm"

        # 2. Segment sizes
        self.portfolio_size = 2048
        self.positions_info_size = 2048

        # 3. Individual locks
        self.portfolio_lock = multiprocessing.Lock()
        self.positions_info_lock = multiprocessing.Lock()

        # Legacy backward-compatibility attributes
        self.shm_name = self.portfolio_name
        self.lock = self.portfolio_lock

    # ── Segment Helpers ────────────────────────────────────────────────

    def _init_segment(self, name: str, size: int, initial_data: dict, lock: multiprocessing.Lock):
        try:
            # Clean up if a previous run didn't exit cleanly
            try:
                old_shm = SharedMemory(name=name)
                old_shm.close()
                old_shm.unlink()
            except Exception:
                pass

            shm = SharedMemory(name=name, create=True, size=size)
            serialized = json.dumps(initial_data).encode('utf-8')
            padded = serialized.ljust(size, b'\x00')
            with lock:
                shm.buf[:size] = padded
            shm.close()
        except Exception as e:
            raise SHMError(f"Failed to initialize segment {name}: {e}")

    def _read_segment(self, name: str, size: int, lock: multiprocessing.Lock) -> dict:
        try:
            shm = SharedMemory(name=name)
            raw = b''
            with lock:
                if shm.buf is None:
                    raise SHMError(f"Shared memory buffer for {name} is unavailable.")
                raw = bytes(shm.buf[:]).split(b'\x00')[0]
            shm.close()
            if not raw:
                return {}
            return json.loads(raw.decode('utf-8'))
        except Exception as e:
            raise SHMError(f"Failed to read segment {name}: {e}")

    def _write_segment(self, name: str, size: int, data: dict, lock: multiprocessing.Lock):
        try:
            serialized = json.dumps(data).encode('utf-8')
            if len(serialized) > size:
                raise SHMError(f"Data size ({len(serialized)} bytes) exceeds segment {name} allocation of {size} bytes.")

            padded_data = serialized.ljust(size, b'\x00')
            shm = SharedMemory(name=name)
            with lock:
                if shm.buf is not None:
                    shm.buf[:size] = padded_data
            shm.close()
        except Exception as e:
            raise SHMError(f"Failed to write to segment {name}: {e}")

    # ── Public Segment API ─────────────────────────────────────────────

    def read_portfolio(self) -> dict:
        return self._read_segment(self.portfolio_name, self.portfolio_size, self.portfolio_lock)

    def write_portfolio(self, portfolio: dict):
        self._write_segment(self.portfolio_name, self.portfolio_size, portfolio, self.portfolio_lock)

    def read_positions_info(self) -> dict:
        return self._read_segment(self.positions_info_name, self.positions_info_size, self.positions_info_lock)

    def write_positions_info(self, positions_info: dict):
        self._write_segment(self.positions_info_name, self.positions_info_size, positions_info, self.positions_info_lock)

    # ── Lifecycle API ──────────────────────────────────────────────────

    def init_shm(self, initial_state: dict):
        portfolio = initial_state.get("portfolio", {"cash": 100000.0, "positions": {}})
        positions_info = initial_state.get("positions_info", {})

        self._init_segment(self.portfolio_name, self.portfolio_size, portfolio, self.portfolio_lock)
        self._init_segment(self.positions_info_name, self.positions_info_size, positions_info, self.positions_info_lock)

    def cleanup(self):
        for name in [self.portfolio_name, self.positions_info_name]:
            try:
                shm = SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except Exception:
                pass

shm_service = SharedMemoryService()
