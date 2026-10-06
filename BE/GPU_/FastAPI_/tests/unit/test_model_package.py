import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from app.trainer.checkpoint_manifest import build_metadata
from app.trainer.model_package import REQUIRED_FILES, find_and_validate_package, validate_package


def valid_metadata() -> dict:
    return {
        "architecture": "DIFFUSION_UNET_TIMM_UMI",
        "framework": "PyTorch",
        "frameworkVersion": "test",
        "contractVersion": "umi-policy-v1",
        "robotProfileId": "52000000-0000-4000-8000-000000000001",
        "actionSpace": "EEF_RELATIVE_ROT6D",
        "actionSpec": {"dim": 10},
        "runtimeSpec": {"execSliceAppliesTo": "raw"},
        "nParams": 35_072_492,
        "controlRateHz": 10,
        "cameraNames": ["camera0_rgb"],
        "inputSchema": {"stateShape": [10]},
        "outputSchema": {"actionShape": [16, 10]},
    }


def create_package(path: Path, *, missing: str | None = None, mode: str = "w") -> None:
    contents = {name: b"test" for name in REQUIRED_FILES}
    contents["metadata.json"] = json.dumps(valid_metadata()).encode()
    contents["diffusion.manifest.json"] = b"{}"
    if missing:
        contents.pop(missing)
    with tarfile.open(path, mode=mode) as archive:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class ModelPackageTest(unittest.TestCase):
    def test_finds_valid_uncompressed_tgz_and_reads_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "model.tgz"
            create_package(package)

            found = find_and_validate_package(Path(directory))
            metadata = build_metadata({}, found)

            self.assertEqual(found, package)
            self.assertEqual(metadata["nParams"], 35_072_492)

    def test_rejects_missing_required_file(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "model.tgz"
            create_package(package, missing="encoder.pt")
            with self.assertRaisesRegex(ValueError, "필수 파일"):
                validate_package(package)

    def test_rejects_gzip_compressed_tgz(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "model.tgz"
            create_package(package, mode="w:gz")
            with self.assertRaisesRegex(ValueError, "압축하지 않은 TAR"):
                validate_package(package)
