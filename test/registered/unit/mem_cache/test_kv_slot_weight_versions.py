import unittest

import torch

from sglang.srt.mem_cache.kv_slot_weight_versions import KvSlotWeightVersions
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _table(num_slots: int = 16) -> KvSlotWeightVersions:
    return KvSlotWeightVersions(num_slots=num_slots, device="cpu")


def _slots(*indices: int) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.int64)


class TestKvSlotWeightVersions(CustomTestCase):
    def test_never_written_slots_report_unknown(self):
        """A fresh table attributes every slot to the unknown version."""
        self.assertEqual(
            _table().lookup_spans(_slots(0, 1, 2)),
            [WeightVersionSpan(version="unknown", start=0, end=3)],
        )

    def test_single_version_collapses_into_one_span(self):
        """Slots written by one version compress into a single span."""
        table = _table()
        table.record(_slots(3, 4, 5), version="v0")

        self.assertEqual(
            table.lookup_spans(_slots(3, 4, 5)),
            [WeightVersionSpan(version="v0", start=0, end=3)],
        )

    def test_rewriting_slots_under_a_new_version_splits_the_lookup(self):
        """Re-recording the tail of a sequence yields an old-version prefix and a new-version suffix."""
        table = _table()
        table.record(_slots(1, 2, 3, 4), version="v0")
        table.record(_slots(3, 4), version="v1")

        self.assertEqual(
            table.lookup_spans(_slots(1, 2, 3, 4)),
            [
                WeightVersionSpan(version="v0", start=0, end=2),
                WeightVersionSpan(version="v1", start=2, end=4),
            ],
        )

    def test_unwritten_slots_interleave_as_unknown_spans(self):
        """A slot that no forward ever wrote breaks a run into three spans."""
        table = _table()
        table.record(_slots(1, 3), version="v0")

        self.assertEqual(
            table.lookup_spans(_slots(1, 2, 3)),
            [
                WeightVersionSpan(version="v0", start=0, end=1),
                WeightVersionSpan(version="unknown", start=1, end=2),
                WeightVersionSpan(version="v0", start=2, end=3),
            ],
        )

    def test_non_adjacent_slots_with_the_same_version_merge(self):
        """Compression follows lookup order, not slot order, so the same version merges."""
        table = _table()
        table.record(_slots(9, 2, 5), version="v0")

        self.assertEqual(
            table.lookup_spans(_slots(9, 2, 5)),
            [WeightVersionSpan(version="v0", start=0, end=3)],
        )

    def test_version_ids_are_interned_and_never_reassigned(self):
        """Re-recording an already seen version reuses its id instead of growing the table."""
        table = _table()
        table.record(_slots(0), version="v0")
        table.record(_slots(1), version="v1")
        table.record(_slots(2), version="v0")

        self.assertEqual(table._version_str_by_id, ["v0", "v1"])
        self.assertEqual(
            table.lookup_spans(_slots(0, 2, 1)),
            [
                WeightVersionSpan(version="v0", start=0, end=2),
                WeightVersionSpan(version="v1", start=2, end=3),
            ],
        )

    def test_empty_lookup_returns_no_spans(self):
        """Looking up an empty prompt yields an empty span list."""
        self.assertEqual(_table().lookup_spans(_slots()), [])


if __name__ == "__main__":
    unittest.main()
