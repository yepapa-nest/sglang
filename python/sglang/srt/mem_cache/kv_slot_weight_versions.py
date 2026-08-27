from __future__ import annotations

from typing import Dict, List

import torch

from sglang.srt.utils.weight_versions import (
    UNKNOWN_WEIGHT_VERSION_ID,
    WeightVersionSpans,
    compress_version_ids_to_spans,
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
