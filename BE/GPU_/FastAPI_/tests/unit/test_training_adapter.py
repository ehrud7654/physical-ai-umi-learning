import unittest

from app.trainer.training_adapter import TrainingAdapter


class TrainingAdapterTest(unittest.TestCase):
    def test_parses_epoch_fraction(self):
        self.assertEqual(TrainingAdapter._parse_epoch("Epoch 7/120 loss=0.1", 120), (7, 120))
        self.assertEqual(TrainingAdapter._parse_epoch("epoch=8 of 120", 120), (8, 120))

    def test_parses_epoch_with_configured_total(self):
        self.assertEqual(TrainingAdapter._parse_epoch("epoch: 14 loss=0.1", 120), (14, 120))

    def test_ignores_non_epoch_log(self):
        self.assertIsNone(TrainingAdapter._parse_epoch("loading dataset", 120))

    def test_run_name_is_filesystem_safe(self):
        self.assertEqual(TrainingAdapter._safe_run_name("job:a/b"), "job_a_b")


if __name__ == "__main__":
    unittest.main()
