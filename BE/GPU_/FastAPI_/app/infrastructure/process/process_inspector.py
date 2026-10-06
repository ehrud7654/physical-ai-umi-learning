import psutil


class ProcessInspector:
    @staticmethod
    def running(pid: int | None) -> bool:
        if pid is None:
            return False
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

