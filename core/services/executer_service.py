import json
import socket
import logging

logger = logging.getLogger(__name__)

class ExecutorService:
    """
    Client-side API wrapper to communicate with the Executer worker process.
    Connects via Unix Domain Socket (UDS) and sends JSON-serialised commands.

    Holds one persistent connection and reuses it across sends instead of
    connecting fresh every call. Connects lazily on first use and
    reconnects automatically the next time send_order() is called if the
    connection was never established or has dropped.
    """
    def __init__(self, socket_path: str = "/tmp/executer.sock"):
        self.socket_path = socket_path
        self._conn = None

    def _ensure_connected(self) -> bool:
        if self._conn is not None:
            return True
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)
            self._conn = sock
            logger.info("[ExecutorService] Connected to %s.", self.socket_path)
            return True
        except (socket.error, FileNotFoundError) as exc:
            logger.debug("[ExecutorService] Connection failed: %s", exc)
            self._conn = None
            return False

    def send_order(self, order: dict) -> bool:
        """
        Send an order dictionary to the Executer over the persistent
        connection (reconnecting first if needed).
        Returns True on success, False otherwise.
        """
        if not self._ensure_connected():
            return False
        try:
            payload = (json.dumps(order) + "\n").encode('utf-8')
            self._conn.sendall(payload)
            logger.info(f"[ExecutorService] Successfully sent order to Executer: {order}")
            return True
        except socket.error as exc:
            logger.error(f"[ExecutorService] Failed to send order: {exc}")
            self.close()
            return False

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def send_raw_command(self, action: str, **kwargs) -> bool:
        """
        Convenience method to construct and send a raw action command.
        """
        cmd = {"action": action, **kwargs}
        return self.send_order(cmd)
