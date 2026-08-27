from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.utils.weight_versions import (
    UNKNOWN_WEIGHT_VERSION_ID,
    WeightVersionSpans,
    compress_version_ids_to_spans,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


def maybe_record_prefill_weight_versions(
    req: Req,
    kv_slot_weight_versions: Optional[KvSlotWeightVersions],
    req_to_token_pool: ReqToTokenPool,
) -> None:
    if kv_slot_weight_versions is None or req.prefill_weight_versions is not None:
        return
    if req.req_pool_idx is None:
        return

    num_prompt_tokens = min(len(req.origin_input_ids), req.kv_committed_len)
    req.prefill_weight_versions = kv_slot_weight_versions.lookup_spans(
        req_to_token_pool.req_to_token[req.req_pool_idx, :num_prompt_tokens]
    )


class KvSlotWeightVersions:
    def __init__(self, *, num_slots: int, device: str):
        self._slot_version_ids = torch.full(
            (num_slots,),
            UNKNOWN_WEIGHT_VERSION_ID,
            dtype=torch.int32,
            device=device,
        )
        self._version_id_by_str: Dict[str, int] = {}
        self._version_str_by_id: List[str] = []

    def record(self, slot_indices: torch.Tensor, version: str) -> None:
        self._slot_version_ids[slot_indices] = self._intern(version=version)

    def lookup_spans(self, slot_indices: torch.Tensor) -> WeightVersionSpans:
        return compress_version_ids_to_spans(
            self._slot_version_ids[slot_indices].tolist(),
            version_str_by_id=self._version_str_by_id,
        )

    def _intern(self, version: str) -> int:
        if (version_id := self._version_id_by_str.get(version)) is not None:
            return version_id

        version_id = len(self._version_str_by_id)
        self._version_str_by_id.append(version)
        self._version_id_by_str[version] = version_id
        return version_id
