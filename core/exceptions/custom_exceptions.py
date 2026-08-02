class BotBaseException(Exception):
    """Base exception class for all custom exceptions in mtbot."""
    pass

class IPCError(BotBaseException):
    """Raised when there is an issue with UDS or connection channels."""
    pass

class SHMError(BotBaseException):
    """Raised when shared memory operations or serialization fail."""
    pass

class ProcessError(BotBaseException):
    """Raised when a process encounters a lifecycle or heartbeat failure."""
    pass
