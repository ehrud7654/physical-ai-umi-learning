import os
import subprocess
import sys
from pathlib import Path


class ProcessLauncher:
    def launch(self, run_id: str, run_directory: Path, gpu_index: int, duration_seconds: int) -> int:
        stdout = (run_directory / "stdout.log").open("ab")
        stderr = (run_directory / "stderr.log").open("ab")
        command = [
            sys.executable, "-m", "app.trainer.entrypoint",
            "--run-id", run_id,
            "--run-directory", str(run_directory),
            "--duration-seconds", str(duration_seconds),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
        stdout.close()
        stderr.close()
        return process.pid

