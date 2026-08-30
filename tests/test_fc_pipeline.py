#!/usr/bin/env python3
"""Testes unitários sem dependências externas para a Fase 1."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fc_common import load_registry, parse_prediction_record, validate_target  # noqa: E402
from generate_fc_dataset import build_dataset  # noqa: E402
from validate_fc_dataset import validate_record  # noqa: E402


class FunctionCallingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(ROOT / "data" / "tools" / "android_tools.json")

    def test_registry_has_ten_tools(self) -> None:
        self.assertEqual(len(self.registry), 10)
        self.assertIn("wifi_set_state", self.registry)
        self.assertIn("alarm_create", self.registry)

    def test_dataset_is_deterministic_and_balanced(self) -> None:
        first = build_dataset(self.registry, 20260830)
        second = build_dataset(self.registry, 20260830)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 720)
        self.assertEqual(sum(r["metadata"]["kind"] == "call" for r in first), 360)
        self.assertEqual(sum(r["metadata"]["kind"] == "abstain" for r in first), 360)
        self.assertEqual(
            {split: sum(r["split"] == split for r in first) for split in ("train", "dev", "test")},
            {"train": 480, "dev": 120, "test": 120},
        )

    def test_generated_records_validate(self) -> None:
        records = build_dataset(self.registry, 20260830)
        for line_number, record in enumerate(records, start=1):
            self.assertEqual(validate_record(record, self.registry, line_number), [])

    def test_invalid_arguments_are_rejected(self) -> None:
        invalid = {"action": "call", "tool": "brightness_set", "arguments": {"level": 101}}
        self.assertTrue(validate_target(invalid, self.registry))
        invalid_extra = {"action": "call", "tool": "wifi_set_state", "arguments": {"enabled": True, "x": 1}}
        self.assertTrue(validate_target(invalid_extra, self.registry))

    def test_qwen_markup_parser(self) -> None:
        raw = "<tool_call><function=wifi_set_state><parameter=enabled>True</parameter></function></tool_call>"
        parsed, is_json, error = parse_prediction_record({"id": "x", "raw": raw})
        self.assertIsNone(error)
        self.assertFalse(is_json)
        self.assertEqual(parsed, {"action": "call", "tool": "wifi_set_state", "arguments": {"enabled": True}})


if __name__ == "__main__":
    unittest.main()
