"""Unit tests for deterministic distillation routing and target formatting.

CALLING SPEC:
    python -m unittest tests.test_distillation_tools

    Inputs/outputs: none; assertions cover pure helpers with fake clients.
    Side effects: none. Does not load models, use GPUs, or call external APIs.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from configs.schema import DistillCfg
from med_pipeline.data.prompts import cot_assistant
from med_pipeline.tools.distill_quality import passes_distill_quality
from med_pipeline.tools import teacher_client


class DistillationToolsTest(unittest.TestCase):
    def test_default_teacher_is_local_qwen_on_eight_gpus(self) -> None:
        cfg = DistillCfg()
        self.assertEqual(cfg.provider, "qwen_sglang")
        self.assertEqual(cfg.teacher_model, "Qwen/Qwen3.6-27B")
        self.assertEqual(cfg.temperature, 0.7)
        self.assertEqual(cfg.server_url, "http://127.0.0.1:30000/v1")

    def test_teacher_registry_routes_qwen_config(self) -> None:
        cfg = DistillCfg()
        sentinel = object()
        with patch.object(teacher_client, "_create_sglang", return_value=sentinel):
            registry = {
                **teacher_client.TEACHER_CLIENTS,
                "qwen_sglang": teacher_client._create_sglang,
            }
            with patch.object(teacher_client, "TEACHER_CLIENTS", registry):
                self.assertIs(teacher_client.create_teacher_client(cfg), sentinel)

    def test_cot_has_exactly_one_final_answer(self) -> None:
        result = cot_assistant("分析过程。\n最终答案：\\boxed{B}", "B")
        self.assertEqual(result, "分析过程。\n\n最终答案：\\boxed{B}")
        self.assertEqual(result.count("\\boxed{"), 1)

    def test_quality_gate_rejects_leaked_meta_reasoning(self) -> None:
        good = "1. 成人侧卧位正常范围为80至180。\n\n最终答案：\\boxed{A}"
        leaked = "Here's a thinking process: Analyze User Input。\\boxed{A}"
        self.assertTrue(passes_distill_quality(good))
        self.assertFalse(passes_distill_quality(leaked))



if __name__ == "__main__":
    unittest.main()
