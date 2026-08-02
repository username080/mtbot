import logging
import traceback

def handle_exception(exc, context_worker=None):
    """
    Centralized handler for exceptions.
    Logs error messages, prints stack traces, and notifies the manager if run inside a worker context.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    trace = traceback.format_exc()
    
    error_msg = f"[EXCEPTION] {exc_type}: {exc_msg}\nTraceback:\n{trace}"
    
    if context_worker:
        # If in a child process worker context, send the error log to the Manager via Redis
        try:
            context_worker.send_log(f"Critical error in worker '{context_worker.worker_name}': {exc_type}: {exc_msg}", errnum=1)
        except Exception as redis_exc:
            print(f"[CRITICAL] Failed to send error log to Redis: {redis_exc}")
            print(error_msg)
    else:
        # If in the Manager process context, print/log directly
        print(f"[CRITICAL] {error_msg}")
        logger = logging.getLogger("QuantSystem")
        if logger.hasHandlers():
            logger.error(f"Critical error: {exc_type} - {exc_msg}")
