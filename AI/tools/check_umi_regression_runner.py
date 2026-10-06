"""Unit checks for the local-only UMI regression runner."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_umi_regression import CHECKS, is_shared_gpu_server


class Tests(unittest.TestCase):
    def test_manifest_has_unique_existing_scripts(self):
        self.assertEqual(len(CHECKS), len(set(CHECKS)))
        root = Path(__file__).resolve().parent
        self.assertTrue(all((root / name).is_file() for name in CHECKS))

    def test_shared_server_is_blocked(self):
        self.assertTrue(is_shared_gpu_server("jupyter-test"))
        self.assertTrue(is_shared_gpu_server("jupyter-test"))
        self.assertFalse(is_shared_gpu_server("DESKTOP-LOCAL"))


if __name__ == "__main__":
    unittest.main()
