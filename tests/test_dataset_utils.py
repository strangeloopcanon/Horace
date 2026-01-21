import random
import unittest


class TestDatasetUtils(unittest.TestCase):
    def test_sample_window_respects_min_max_chars(self):
        from tools.studio.dataset_utils import sample_window

        text = ("Para one.\n" * 200) + "\n\n" + ("Para two.\n" * 200)
        rng = random.Random(0)
        out = sample_window(text, rng=rng, max_chars=1000, min_chars=700)
        self.assertGreaterEqual(len(out), 700)
        self.assertLessEqual(len(out), 1000)


if __name__ == "__main__":
    unittest.main()

