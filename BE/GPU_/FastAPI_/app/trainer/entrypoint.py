import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.trainer.progress_reporter import ProgressReporter
from app.trainer.training_adapter import TrainingAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, required=True)
    args = parser.parse_args()

    request = json.loads((args.run_directory / "request.json").read_text(encoding="utf-8"))
    reporter = ProgressReporter(args.run_directory / "progress.json")
    checkpoint_relative = Path("checkpoints") / request["trainingJobId"] / str(request["attempt"]) / "model.tgz"
    shared_root = args.run_directory.parents[2]
    checkpoint = shared_root / checkpoint_relative
    try:
        adapter = TrainingAdapter()
        result = adapter.train(request, checkpoint, reporter, args.duration_seconds)
        result["checkpointStorageKey"] = checkpoint_relative.as_posix()
        completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        reporter.report(
            "COMPLETED", "CHECKPOINT_READY", 100,
            epoch=int(os.getenv("TRAINING_EPOCHS", "120")),
            totalEpochs=int(os.getenv("TRAINING_EPOCHS", "120")),
            modelCompletedAt=completed_at, result=result,
        )
    except BaseException as error:
        reporter.report("FAILED", "FAILED", 0, error={"code": "TRAINING_PROCESS_FAILED", "message": str(error)})
        raise


if __name__ == "__main__":
    main()

