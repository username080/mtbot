import os
import subprocess
from manager.manager import Manager
from core.redis_c import RedisStarter

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # In Docker, Redis runs as a sidecar container — no need to start a local process.
    # When REDIS_HOST is set to anything other than localhost, we skip the local redis-server.
    redis_host = os.getenv("REDIS_HOST", "localhost")
    if redis_host == "localhost":
        redis_starter = RedisStarter()
        redis_starter.start()
    else:
        print(f"[INFO] REDIS_HOST={redis_host!r} — skipping local redis-server, using external Redis.")
        redis_starter = None

    # Start the Web UI (FastAPI + uvicorn) as a subprocess
    web_port = os.getenv("WEB_PORT", "6594")
    redis_port = os.getenv("REDIS_PORT", "6379")
    print(f"Starting Web UI on 0.0.0.0:{web_port}...")
    ui_process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "web.app:app",
         "--host", "0.0.0.0", "--port", web_port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    app_manager = Manager()
    try:
        app_manager.start()
    except KeyboardInterrupt:
        print("Server is being SHUTDOWN manually...")
    finally:
        print("Shutting down Web UI...")
        ui_process.terminate()
        ui_process.wait()
        if redis_starter:
            redis_starter.stop()