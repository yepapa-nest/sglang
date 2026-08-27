import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-small")

_REQUEST_TIMEOUT = 180

_SHARED_PREFIX = (
    "You are a meticulous assistant. Follow every instruction to the letter, "
    "explain your reasoning step by step, and never skip an intermediate result. "
    "Here is the background material you must rely on for the whole conversation. "
)


def _assert_prefill_spans_contiguous(test, meta_info):
    spans = meta_info["prefill_weight_versions"]
    test.assertGreater(len(spans), 0)
    test.assertEqual(spans[0]["start"], 0)
    test.assertGreater(spans[-1]["end"], 0)
    for prev, cur in zip(spans, spans[1:]):
        test.assertEqual(prev["end"], cur["start"])
        test.assertNotEqual(prev["version"], cur["version"])
    if "prompt_tokens" in meta_info:
        test.assertEqual(spans[-1]["end"], meta_info["prompt_tokens"])
    return spans


class _PrefillWeightVersionServerMixin:
    @classmethod
    def _launch(cls, other_args):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt: str, max_new_tokens: int = 8):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _flush_cache(self):
        requests.post(f"{self.base_url}/flush_cache", timeout=30).raise_for_status()

    def _current_version(self):
        response = requests.get(f"{self.base_url}/get_model_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        return response.json()["weight_version"]

    def _pause(self, mode: str):
        requests.post(
            f"{self.base_url}/pause_generation", json={"mode": mode}, timeout=30
        ).raise_for_status()

    def _continue(self):
        requests.post(
            f"{self.base_url}/continue_generation", json={}, timeout=30
        ).raise_for_status()

    def _abort_all(self):
        requests.post(
            f"{self.base_url}/abort_request", json={"abort_all": True}, timeout=30
        ).raise_for_status()

    def _set_weight_version(self, new_version: str):
        response = requests.post(
            f"{self.base_url}/update_weight_version",
            json={"new_version": new_version, "abort_all_requests": False},
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)


class TestPrefillWeightVersionSpans(_PrefillWeightVersionServerMixin, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._launch(
            [
                "--weight-version",
                "prefill-v0",
                "--enable-prefill-weight-versions",
            ]
        )

    def test_01_cold_prompt_reports_a_single_span(self):
        """A prompt computed entirely under one version reports one span over the whole prompt."""
        self._flush_cache()
        data = self._generate(_SHARED_PREFIX + "Case one: count to three.")

        meta_info = data["meta_info"]
        self.assertEqual(meta_info["cached_tokens"], 0)
        spans = _assert_prefill_spans_contiguous(self, meta_info)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], "prefill-v0")

    def test_02_reused_prefix_keeps_the_version_that_computed_it(self):
        """A radix hit on KV computed before a relabel shows up as a stale leading span."""
        self._flush_cache()
        self._generate(_SHARED_PREFIX + "Case two: name three colors.")

        self._set_weight_version("prefill-v1")
        self.assertEqual(self._current_version(), "prefill-v1")
        data = self._generate(
            _SHARED_PREFIX + "Case two: name three colors. Then name three shapes."
        )

        meta_info = data["meta_info"]
        self.assertGreater(meta_info["cached_tokens"], 0)
        spans = _assert_prefill_spans_contiguous(self, meta_info)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["version"], "prefill-v0")
        self.assertEqual(spans[0]["end"], meta_info["cached_tokens"])
        self.assertEqual(spans[1]["version"], "prefill-v1")

    def test_03_relabel_during_decode_leaves_the_prompt_spans_alone(self):
        """Prompt attribution follows the KV, so a mid-generation relabel only moves the sampling spans."""
        self._flush_cache()
        version_before = self._current_version()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._generate,
                prompt=_SHARED_PREFIX + "Case four: write a long story.",
                max_new_tokens=256,
            )
            time.sleep(2)
            self._pause("in_place")
            try:
                self._set_weight_version("prefill-v2")
            finally:
                self._continue()
            data = future.result()

        meta_info = data["meta_info"]
        spans = _assert_prefill_spans_contiguous(self, meta_info)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], version_before)
        self.assertEqual(meta_info["weight_versions"][0]["version"], version_before)

    def test_04_aborted_requests_still_report_prompt_spans(self):
        """An aborted request keeps the prompt spans recorded when its prefill finished."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._generate,
                prompt=_SHARED_PREFIX + "Case four: write an even longer story.",
                max_new_tokens=8192,
            )
            time.sleep(1)
            self._pause("in_place")
            try:
                self._abort_all()
            finally:
                self._continue()
            data = future.result()

        meta_info = data["meta_info"]
        self.assertEqual(meta_info["finish_reason"]["type"], "abort")
        self.assertGreater(meta_info["completion_tokens"], 0)
        spans = _assert_prefill_spans_contiguous(self, meta_info)
        self.assertEqual(spans[-1]["version"], self._current_version())

    def test_05_requests_aborted_before_prefill_carry_no_prompt_spans(self):
        """A request aborted before its prefill ran has no prompt KV to attribute."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            self._pause("in_place")
            try:
                future = executor.submit(
                    self._generate,
                    prompt=_SHARED_PREFIX + "Case five: say hello.",
                    max_new_tokens=8,
                )
                time.sleep(1)
                self._abort_all()
            finally:
                self._continue()
            data = future.result()

        meta_info = data["meta_info"]
        self.assertEqual(meta_info["finish_reason"]["type"], "abort")
        self.assertEqual(meta_info["completion_tokens"], 0)
        self.assertNotIn("prefill_weight_versions", meta_info)

    def test_06_openai_metadata_contains_prefill_weight_versions(self):
        """OpenAI-compatible responses surface the prompt spans next to the sampling spans."""
        self._flush_cache()
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": _SHARED_PREFIX + "Case six: say hello.",
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)

        metadata = response.json()["metadata"]
        spans = metadata["prefill_weight_versions"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], self._current_version())
        self.assertEqual(spans[0]["start"], 0)
        self.assertIn("weight_versions", metadata)


if __name__ == "__main__":
    unittest.main()
