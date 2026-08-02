"""
Generic message dispatch, mixed into BaseWorker.

Instead of each worker hand-decoding its own inbound messages with a
scattered if/elif chain per channel (which is what check_broadcast() used
to do, and what Executer's/DataPipeline's UDS connection-read loops still
do today - migrated in a later step), a worker registers one handler
function per MPacket `type` it cares about, and calls dispatch() whenever a
decoded MPacket arrives from ANY channel (Redis pub/sub today; UDS sockets
once those channels migrate too). This class doesn't know or care what
transport the message came in on - only MPacket.type -> handler.
"""
from typing import Callable, Dict

from core.packets.m_packet import MPacket


class MessageDispatcherMixin:
    """
    Provides register_handler()/dispatch(). Requires the including class to:
      - set `self._handlers = {}` before either method is used (BaseWorker
        does this in __init__)
      - provide a `send_log(message, errnum=0)` method, so dispatch
        failures get reported the same way any other worker error does
        (BaseWorker already has this)
    """

    _handlers: Dict[str, Callable[[MPacket], None]]

    def register_handler(self, msg_type: str, handler: Callable[[MPacket], None]) -> None:
        self._handlers[msg_type] = handler

    def dispatch(self, packet: MPacket) -> None:
        handler = self._handlers.get(packet.type)
        if handler is None:
            self.send_log(
                f"No handler registered for message type '{packet.type}' (from {packet.sender})",
                errnum=1,
            )
            return
        try:
            handler(packet)
        except Exception as exc:
            # A malformed message should never be able to kill the whole
            # worker process - see test_crash_supervision.py, which
            # documented a ValueError from a bad order field doing exactly
            # that before this existed.
            self.send_log(
                f"Handler for '{packet.type}' raised {type(exc).__name__}: {exc}",
                errnum=1,
            )
