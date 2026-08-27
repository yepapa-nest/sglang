import unittest
from typing import List

import torch

from sglang.srt.mem_cache.kv_slot_weight_versions import (
    KvSlotWeightVersions,
    maybe_record_prefill_weight_versions,
)
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _table(num_slots: int = 16) -> KvSlotWeightVersions:
    return KvSlotWeightVersions(num_slots=num_slots, device="cpu")


def _slots(*indices: int) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.int64)


class _ReqStub:
    def __init__(self, num_prompt_tokens: int, kv_committed_len: int):
        self.origin_input_ids = [0] * num_prompt_tokens
        self.kv_committed_len = kv_committed_len
        self.req_pool_idx = 1
        self.prefill_weight_versions = None


class _ReqToTokenPoolStub:
    def __init__(self, slots_of_req: List[int]):
        self.req_to_token = torch.zeros((2, 32), dtype=torch.int32)
        self.req_to_token[1, : len(slots_of_req)] = torch.tensor(
            slots_of_req, dtype=torch.int32
        )


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


class TestMaybeRecordPrefillWeightVersions(CustomTestCase):
    def test_prompt_slots_are_looked_up_and_stored_on_the_request(self):
        """The prompt's KV slots resolve to the versions that computed them."""
        table = _table()
        table.record(_slots(4, 5), version="v0")
        table.record(_slots(6), version="v1")
        req = _ReqStub(num_prompt_tokens=3, kv_committed_len=5)

        maybe_record_prefill_weight_versions(
            req,
            kv_slot_weight_versions=table,
            req_to_token_pool=_ReqToTokenPoolStub([4, 5, 6, 7, 8]),
        )

        self.assertEqual(
            req.prefill_weight_versions,
            [
                WeightVersionSpan(version="v0", start=0, end=2),
                WeightVersionSpan(version="v1", start=2, end=3),
            ],
        )

    def test_lookup_is_clamped_to_the_committed_kv_length(self):
        """A prompt aborted mid-prefill reports only the tokens whose KV was actually written."""
        table = _table()
        table.record(_slots(4, 5), version="v0")
        req = _ReqStub(num_prompt_tokens=5, kv_committed_len=2)

        maybe_record_prefill_weight_versions(
            req,
            kv_slot_weight_versions=table,
            req_to_token_pool=_ReqToTokenPoolStub([4, 5]),
        )

        self.assertEqual(
            req.prefill_weight_versions,
            [WeightVersionSpan(version="v0", start=0, end=2)],
        )

    def test_disabled_tracking_leaves_the_request_untouched(self):
        """With no table the request keeps None, so nothing reaches meta_info."""
        req = _ReqStub(num_prompt_tokens=3, kv_committed_len=3)

        maybe_record_prefill_weight_versions(
            req,
            kv_slot_weight_versions=None,
            req_to_token_pool=_ReqToTokenPoolStub([4, 5, 6]),
        )

        self.assertIsNone(req.prefill_weight_versions)

    def test_an_already_recorded_request_is_not_looked_up_again(self):
        """The first lookup wins, so a later abort cannot overwrite freed-slot garbage in."""
        table = _table()
        table.record(_slots(4, 5, 6), version="v1")
        req = _ReqStub(num_prompt_tokens=3, kv_committed_len=3)
        req.prefill_weight_versions = [
            WeightVersionSpan(version="v0", start=0, end=3)
        ]

        maybe_record_prefill_weight_versions(
            req,
            kv_slot_weight_versions=table,
            req_to_token_pool=_ReqToTokenPoolStub([4, 5, 6]),
        )

        self.assertEqual(
            req.prefill_weight_versions,
            [WeightVersionSpan(version="v0", start=0, end=3)],
        )

    def test_a_request_without_a_pool_slot_is_skipped(self):
        """A request whose KV was never allocated has nothing to look up."""
        req = _ReqStub(num_prompt_tokens=3, kv_committed_len=3)
        req.req_pool_idx = None

        maybe_record_prefill_weight_versions(
            req,
            kv_slot_weight_versions=_table(),
            req_to_token_pool=_ReqToTokenPoolStub([4, 5, 6]),
        )

        self.assertIsNone(req.prefill_weight_versions)


if __name__ == "__main__":
    unittest.main()
