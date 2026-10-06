from pathlib import Path
import json
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app/src"))

from umi_preprocessor.contracts import extract_archive, validate_name  # noqa: E402
from umi_preprocessor.packaging import package_training_input  # noqa: E402


def main() -> None:
    assert validate_name("pick-001", "jobId") == "pick-001"
    try:
        validate_name("../escape", "jobId")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe job id accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        (source / "hello.txt").write_text("ok", encoding="utf-8")
        archive = root / "input.tgz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(source / "hello.txt", arcname="hello.txt")
        extracted = root / "extracted"
        extract_archive(archive, extracted)
        assert (extracted / "hello.txt").read_text(encoding="utf-8") == "ok"

        dataset = root / "sample.zarr.zip"
        dataset.write_bytes(b"zarr fixture")
        dataset.with_suffix(".report.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
        result = package_training_input(dataset)
        assert Path(result["archive"]).is_file()
        with tarfile.open(result["archive"], "r:gz") as bundle:
            assert set(bundle.getnames()) == {
                "sample.zarr.zip", "sample.zarr.report.json", "sample_gpu_input.manifest.json"
            }
    print("PACKAGE_SELF_TEST_OK")


if __name__ == "__main__":
    main()
