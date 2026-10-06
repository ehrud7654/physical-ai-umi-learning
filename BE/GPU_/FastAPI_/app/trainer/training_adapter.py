import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.infrastructure.storage.checksum import sha256_file
from app.trainer.checkpoint_manifest import build_metadata
from app.trainer.model_package import find_and_validate_package
from app.trainer.progress_reporter import ProgressReporter


class TrainingAdapter:
    """검증된 Zarr 입력을 공식 UMI trainer에 전달하는 어댑터."""

    _EPOCH_PATTERNS = (
        re.compile(r"(?:epoch|Epoch)\s*[=: ]\s*(\d+)\s*(?:/|of)\s*(\d+)"),
        re.compile(r"\[(\d+)\s*/\s*(\d+)\].*(?:epoch|Epoch)"),
    )

    def train(
        self,
        request: dict[str, Any],
        output: Path,
        reporter: ProgressReporter,
        duration_seconds: int,
    ) -> dict[str, Any]:
        del duration_seconds  # 실제 trainer가 실행 시간을 결정한다.

        shared_root = Path(os.getenv("SHARED_STORAGE_ROOT", "/data/umi-gpu")).resolve()
        bundle = Path(os.getenv("UMI_TRAINING_BUNDLE_ROOT", "/opt/umi-training")).resolve()
        python = os.getenv("UMI_TRAINING_PYTHON", sys.executable)
        trainer = bundle / "03_umi_policy_trainer" / "train_policy.py"
        profile = Path(
            os.getenv(
                "UMI_TRAINING_PROFILE",
                str(bundle / "03_umi_policy_trainer" / "configs" / "policy_resnet18_gpu.yaml"),
            )
        ).resolve()
        zarr = self._shared_file(shared_root, request["trainingInput"]["storageKey"])
        runs_dir = self._shared_directory(
            shared_root, os.getenv("UMI_TRAINING_RUNS_DIR", "trainer/runs")
        )
        epochs = self._positive_int("TRAINING_EPOCHS", 60)
        batch = self._positive_int("TRAINING_BATCH_SIZE", 8)
        eval_batch = self._positive_int("TRAINING_EVAL_BATCH_SIZE", batch)
        gpu = os.getenv("CUDA_VISIBLE_DEVICES", str(request.get("gpuIndex", 0)))
        run_name = os.getenv("TRAINING_RUN_NAME", request["trainingJobId"])
        # trainingJobId + attempt is stable for a FastAPI run and unique across
        # EC2 retries. Wall-clock based ids can collide and cannot be recovered.
        run_id = f"{self._safe_run_name(run_name)}_attempt_{int(request['attempt'])}"

        self._require_file(trainer, "UMI trainer")
        self._require_file(profile, "UMI trainer profile")
        self._require_directory(umi_root := Path(
            os.getenv("UMI_ROOT", str(bundle / "third_party" / "umi"))
        ).resolve(), "UMI_ROOT")
        runs_dir.mkdir(parents=True, exist_ok=True)

        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "UMI_ROOT": str(umi_root),
                "WANDB_MODE": os.getenv("WANDB_MODE", "disabled"),
                "HYDRA_FULL_ERROR": os.getenv("HYDRA_FULL_ERROR", "1"),
            }
        )
        # An empty seed means "use the official trainer default", not integer zero.
        if not environment.get("TRAINING_SEED", "").strip():
            environment.pop("TRAINING_SEED", None)

        print(f"run-id {run_id} · GPU {gpu}", flush=True)
        reporter.report("RUNNING", "DATASET_CHECK", 5, epoch=0, totalEpochs=epochs)
        self._run(
            [python, str(trainer), "check", str(zarr)],
            cwd=bundle,
            env=environment,
            title="dataset check",
            reporter=reporter,
            stage="DATASET_CHECK",
            epochs=epochs,
        )

        reporter.report("RUNNING", "TRAINING", 10, epoch=0, totalEpochs=epochs)
        self._run(
            [
                python,
                str(trainer),
                "train",
                str(zarr),
                "--run-id",
                run_id,
                "--runs-dir",
                str(runs_dir),
                "--epochs",
                str(epochs),
                "--batch",
                str(batch),
                "--eval-batch",
                str(eval_batch),
                "--profile",
                str(profile),
            ],
            cwd=bundle,
            env=environment,
            title=f"학습 {epochs} epoch",
            reporter=reporter,
            stage="TRAINING",
            epochs=epochs,
        )

        reporter.report(
            "RUNNING", "CHECKPOINT_CREATING", 95, epoch=epochs, totalEpochs=epochs
        )
        trained_package = find_and_validate_package(runs_dir / run_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_package, output)

        return {
            "checkpointStorageKey": str(output).replace("\\", "/"),
            "sizeBytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "metadata": build_metadata(request, output),
        }

    def _run(
        self,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        title: str,
        reporter: ProgressReporter,
        stage: str,
        epochs: int,
    ) -> None:
        print(f"[{title}] {' '.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if stage == "TRAINING":
                    progress = self._parse_epoch(line, epochs)
                    if progress is not None:
                        epoch, total = progress
                        percent = min(94, 10 + int(84 * epoch / max(total, 1)))
                        reporter.report(
                            "RUNNING", "TRAINING", percent,
                            epoch=epoch, totalEpochs=total,
                        )
            return_code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
        if return_code != 0:
            raise RuntimeError(f"{title} 실패: exitCode={return_code}")

    @classmethod
    def _parse_epoch(cls, line: str, configured_epochs: int) -> tuple[int, int] | None:
        for pattern in cls._EPOCH_PATTERNS:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                total = int(match.group(2))
                if epoch >= 0 and total > 0:
                    return min(epoch, total), total
        # Some Hydra/PyTorch trainers log only the current epoch.
        match = re.search(r"(?:epoch|Epoch)\s*[=: ]\s*(\d+)\b", line)
        if match:
            return min(int(match.group(1)), configured_epochs), configured_epochs
        return None

    @staticmethod
    def _shared_file(root: Path, storage_key: str) -> Path:
        path = (root / storage_key).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError("Zarr 입력이 공유 저장소의 일반 파일이 아닙니다.")
        return path

    @staticmethod
    def _shared_directory(root: Path, value: str) -> Path:
        configured = Path(value)
        path = (configured if configured.is_absolute() else root / configured).resolve()
        if not path.is_relative_to(root):
            raise ValueError("학습 결과 경로가 공유 저장소를 벗어났습니다.")
        return path

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다: {path}")

    @staticmethod
    def _require_directory(path: Path, label: str) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} 디렉터리를 찾을 수 없습니다: {path}")

    @staticmethod
    def _positive_int(name: str, default: int) -> int:
        value = int(os.getenv(name, str(default)))
        if value < 1:
            raise ValueError(f"{name}은 1 이상이어야 합니다.")
        return value

    @staticmethod
    def _safe_run_name(value: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in value
        )
        if not safe:
            raise ValueError("학습 run 이름이 올바르지 않습니다.")
        return safe

