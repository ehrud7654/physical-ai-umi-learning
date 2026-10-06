import os
import signal
import time

import psutil


class ProcessController:
    def terminate(self, pid: int | None, grace_seconds: float = 5.0) -> None:
        if pid is None:
            return
        try:
            if os.name == "nt":
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                _, alive = psutil.wait_procs([parent, *children], timeout=grace_seconds)
                for process in alive:
                    process.kill()
                _, alive = psutil.wait_procs(alive, timeout=grace_seconds)
                if alive:
                    raise RuntimeError("Training processes did not exit after cancellation")
            else:
                # The launcher creates a session with pgid == pid. Keep using that
                # group even if the leader exits before its training children.
                group = pid
                os.killpg(group, signal.SIGTERM)
                deadline = time.monotonic() + grace_seconds
                while time.monotonic() < deadline:
                    os.killpg(group, 0)
                    time.sleep(0.1)
                os.killpg(group, signal.SIGKILL)
        except (psutil.NoSuchProcess, ProcessLookupError):
            return

