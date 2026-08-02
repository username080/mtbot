"""
DataPipelineService — Client-side IPC wrapper.

Instantiated inside the Analyser process. The DataPipeline worker runs in a
separate process and exposes a Unix Domain Socket (UDS) server. This class
is the only interface Analyser uses to reach the pipeline - and it's
receive-only: Analyser never sends anything to DataPipeline. There is no
outbound protocol here (no subscribe/unsubscribe/fetch_historical) - the
pipeline just streams its fixed DEFAULT_COINS symbol set unconditionally
for as long as it runs.

Protocol
--------
All messages are newline-delimited JSON.

Incoming (pipeline -> analyser):
  {"type": "dollar_bar", "data": {...}}   — completed Dollar Bar
"""

import json
import socket
import logging

logger = logging.getLogger(__name__)


class DataPipelineService:
    """
    Client-side API wrapper to communicate with the DataPipeline service.
    Instantiated inside the Analyser process.

    Parameters
    ----------
    socket_path : str  path to the Unix Domain Socket exposed by the pipeline
    """

    def __init__(self, socket_path: str = "/tmp/data_pipeline_service.sock"):
        self.socket_path = socket_path
        self._conn   = None
        self._buffer = ""

    # ── Connection lifecycle ───────────────────────────────────────────

    def connect(self) -> bool:
        """
        Establish a persistent non-blocking connection to the pipeline server.

        Returns True on success, False if the server is not yet running.
        """
        if self._conn is not None:
            return True
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)
            sock.setblocking(False)
            self._conn   = sock
            self._buffer = ""
            logger.info("[DataPipelineService] Connected to %s.", self.socket_path)
            return True
        except (socket.error, FileNotFoundError) as exc:
            logger.debug("[DataPipelineService] Connection failed: %s", exc)
            self._conn = None
            return False

    def is_connected(self) -> bool:
        return self._conn is not None

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn   = None
            self._buffer = ""
        logger.info("[DataPipelineService] Connection closed.")

    # ── Streaming read ─────────────────────────────────────────────────

    def read_bars(self) -> list:
        """
        Non-blocking read of all available messages from the pipeline.

        Returns
        -------
        list of dicts, each with keys:
          "type" : "dollar_bar"
          "data" : dict  (bar fields)

        Call this on every event-loop tick in the Analyser.
        """
        packets = []
        if self._conn is None:
            return packets

        try:
            raw = self._conn.recv(65536)
            if not raw:
                logger.warning("[DataPipelineService] Server closed connection.")
                self.close()
                return packets
            self._buffer += raw.decode("utf-8", errors="replace")
        except BlockingIOError:
            pass
        except socket.error as exc:
            logger.warning("[DataPipelineService] Socket error: %s", exc)
            self.close()
            return packets

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                packets.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.debug("[DataPipelineService] JSON parse error: %s | %s", exc, line[:80])

        return packets

    def read_dollar_bars(self) -> list:
        """Returns only completed Dollar Bar packets."""
        return [p["data"] for p in self.read_bars() if p.get("type") == "dollar_bar"]
