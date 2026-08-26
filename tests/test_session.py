"""Tests for the versioned, data-only JSON session format."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from domstudio.services.session import Session, TraceRecord, load_session, save_session


class SessionTests(unittest.TestCase):
    def test_json_roundtrip_preserves_application_state(self) -> None:
        session = Session(
            image_path="survey/outcrop.tif",
            traces=[
                TraceRecord(
                    points=[(1.25, 2.5), (8.0, 13.0)],
                    confidence=0.91,
                    properties={"set": "NE", "reviewed": True},
                )
            ],
            detector={"profile": "analytical", "scales": [0.8, 1.4]},
            trace_settings={"orientation_weight": 0.8},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "project.geotrace.json"
            saved_path = save_session(path, session)
            raw = saved_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            restored = load_session(saved_path)

        self.assertTrue(raw.lstrip().startswith("{"))
        self.assertEqual(payload["version"], 1)
        self.assertEqual(restored.image_path, session.image_path)
        self.assertEqual(len(restored.traces), 1)
        self.assertEqual(restored.traces[0].points, [[1.25, 2.5], [8.0, 13.0]])
        self.assertAlmostEqual(restored.traces[0].confidence or 0.0, 0.91)
        self.assertEqual(restored.traces[0].properties["set"], "NE")
        self.assertEqual(restored.detector["profile"], "analytical")

    def test_rejects_unknown_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.json"
            path.write_text('{"version": 999, "traces": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported session version"):
                load_session(path)

    def test_pickle_or_executable_text_is_not_accepted_as_a_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.session"
            path.write_bytes(b"cos\nsystem\n(S'echo unsafe'\ntR.")
            with self.assertRaises((UnicodeDecodeError, json.JSONDecodeError)):
                load_session(path)


if __name__ == "__main__":
    unittest.main()

