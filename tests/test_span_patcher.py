from __future__ import annotations

import unittest

from tools.studio.span_patcher import _dead_zone_penalty_and_reasons


class TestSpanPatcherDeadZones(unittest.TestCase):
    def test_flat_but_clear_does_not_trigger(self) -> None:
        zone = "He went home. He ate dinner. He slept."
        keep, _pen, reasons = _dead_zone_penalty_and_reasons(zone, sent_len_cv=0.0)
        self.assertFalse(keep)
        self.assertIn("flat_cadence", reasons)

    def test_repetition_triggers(self) -> None:
        zone = "We went home. We went home. We went home. We went home."
        keep, _pen, reasons = _dead_zone_penalty_and_reasons(zone, sent_len_cv=0.0)
        self.assertTrue(keep)
        self.assertIn("flat_cadence", reasons)
        self.assertTrue(any(r.endswith("repetition") for r in reasons))

    def test_droning_density_triggers(self) -> None:
        zone = (
            "The implementation of the optimization strategy was undertaken in order to "
            "facilitate the utilization of available resources."
        )
        keep, _pen, reasons = _dead_zone_penalty_and_reasons(zone, sent_len_cv=None)
        self.assertTrue(keep)
        self.assertIn("droning_density", reasons)

