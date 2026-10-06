"""Local regression tests for immutable UMI raw provenance metadata."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.convert_umi import raw_bundle_digest


class Tests(unittest.TestCase):
    def test_digest_covers_array_metadata_and_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "one.raw.npz"
            meta = root / "one.raw.json"
            npz.write_bytes(b"array-v1")
            meta.write_text('{"id":"one"}', encoding="utf-8")
            original = raw_bundle_digest(root, [npz])
            meta.write_text('{"id":"changed"}', encoding="utf-8")
            self.assertNotEqual(original, raw_bundle_digest(root, [npz]))
            meta.write_text('{"id":"one"}', encoding="utf-8")
            npz.write_bytes(b"array-v2")
            self.assertNotEqual(original, raw_bundle_digest(root, [npz]))

    def test_missing_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "one.raw.npz"
            npz.write_bytes(b"array")
            with self.assertRaises(FileNotFoundError):
                raw_bundle_digest(root, [npz])


if __name__ == "__main__":
    unittest.main()
