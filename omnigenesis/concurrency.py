import threading
from typing import Optional


# Lock protecting all model forward/backward/optimizer operations.
model_lock = threading.Lock()
# Set by the inference thread when the user types 'exit'.
shutdown_event = threading.Event()
# Upper bound for graceful background-thread shutdown waits.
SHUTDOWN_WAIT_SECONDS = 30.0


def wait_for_training_thread(
    train_thread: Optional[threading.Thread], timeout_s: float = SHUTDOWN_WAIT_SECONDS
) -> bool:
    if train_thread is None or not train_thread.is_alive():
        return True
    print("[Main] Waiting for training thread to save checkpoint...", flush=True)
    train_thread.join(timeout=timeout_s)
    if train_thread.is_alive():
        print(
            f"[Main] Training thread did not exit within {timeout_s:.0f}s; forcing process exit.",
            flush=True,
        )
        return False
    return True
