import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from umi.arm_preflight import ArmCommand
from umi.chunk_runtime import Snapshot, run_chunks


class Tests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.calls, self.sent, self.stops = [], [], 0
        self.fail = False
        self.delay = 0.
    def sleep(self, seconds): self.now += seconds
    def acquire(self):
        self.calls.append(self.now)
        return Snapshot(self.now, None)
    def plan(self, payload):
        self.now += self.delay
        return [ArmCommand(np.zeros(5), .045) for _ in range(4)]
    def send(self, arm, width):
        self.sent.append(self.now)
        return not self.fail
    def stop(self): self.stops += 1
    def run_loop(self, acquire=None, plan=None):
        return run_chunks(acquire or self.acquire, plan or self.plan, self,
            cycles=2, n_steps=2, period_s=.1, max_age_s=.5,
            max_lateness_s=.02, clock=lambda:self.now, sleep=self.sleep)
    def test_n_step_replanning(self):
        self.assertEqual(self.run_loop(), 4)
        np.testing.assert_allclose(self.sent, [0, .1, .2, .3])
        np.testing.assert_allclose(self.calls, [0, .2])
        self.assertEqual(self.stops, 0)
    def test_stale_before_inference(self):
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.run_loop(acquire=lambda:Snapshot(-1, None))
        self.assertEqual(self.sent, [])
        self.assertEqual(self.stops, 1)
    def test_slow_inference_no_catchup(self):
        self.delay = .03
        with self.assertRaisesRegex(ValueError, 'deadline'): self.run_loop()
        self.assertEqual(self.sent, [])
    def test_driver_failure_not_retried(self):
        self.fail = True
        with self.assertRaises(RuntimeError): self.run_loop()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.stops, 1)
    def test_inference_exception_stops(self):
        def bad(_): raise RuntimeError('inference failed')
        with self.assertRaisesRegex(RuntimeError, 'inference'): self.run_loop(plan=bad)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.stops, 1)
    def test_short_chunk(self):
        with self.assertRaisesRegex(ValueError, 'shorter'): self.run_loop(plan=lambda _:[])
        self.assertEqual(self.sent, [])


if __name__ == '__main__': unittest.main()
