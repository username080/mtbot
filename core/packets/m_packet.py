import json
import datetime
from typing import Optional

class MPacket:
    """
    Standardized message envelope for all inter-process communication.

    `sender`/`to` identify which process created the message and which
    process it's addressed to ("Manager" | "ALL" for broadcast | a
    worker_name). Note: the field is named `sender`, not `from` - `from` is
    a reserved Python keyword (`from x import y`), so `self.from = ...` is
    a SyntaxError and can't be used as an attribute name at all.

    `msg` stays a human-readable string (what log messages use); `payload`
    carries structured data (order fields, bar fields, symbol lists, ...)
    that used to get stuffed into ad-hoc dicts on each channel.
    """
    def __init__(self, sender: str, to: str, timestamp: str, msg_type: str,
                 errornum: int, msg: str, payload: Optional[dict] = None):
        self.sender = sender
        self.to = to
        self.timestamp = timestamp
        self.type = msg_type
        self.errornum = errornum
        self.msg = msg
        self.payload = payload if payload is not None else {}

    @classmethod
    def create(cls, sender: str, to: str, msg_type: str, errornum: int = 0,
               msg: str = "", payload: Optional[dict] = None):
        timestamp = datetime.datetime.now().isoformat()
        return cls(sender, to, timestamp, msg_type, errornum, msg, payload)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "to": self.to,
            "timestamp": self.timestamp,
            "type": self.type,
            "errornum": self.errornum,
            "msg": self.msg,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            sender=data.get("sender", "Unknown"),
            to=data.get("to", "Unknown"),
            timestamp=data.get("timestamp", ""),
            msg_type=data.get("type", "unknown"),
            errornum=data.get("errornum", 0),
            msg=data.get("msg", ""),
            payload=data.get("payload", {}),
        )

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))
