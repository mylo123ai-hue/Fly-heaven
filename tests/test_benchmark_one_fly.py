from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.benchmark_one_fly import (
    BenchmarkError,
    MAX_DURATION_US,
    Upstream,
    _base_report,
    build_command,
    duration_to_us,
    missing_data_inputs,
    validate_upstream_summary,
)


def upstream_fixture() -> Upstream:
    root = Path("/upstream")
    return Upstream(
        root=root,
        commit="a" * 40,
        scenario_path=root / "configs/scenarios/swarm3d-showcase.json",
        contract_path=root / "configs/experiments/demo01-visual-operating-point-v1.json",
        runner_path=root / "scripts/run_swarm3d_showcase.py",
        coupling_us=15_000,
        scenario_sha256="b" * 64,
        contract_sha256="c" * 64,
        uv_lock_sha256="d" * 64,
        scenario_fly_count=12,
    )


class BenchmarkContractTests(unittest.TestCase):
    def test_duration_is_positive_bounded_and_aligned(self) -> None:
        self.assertEqual(duration_to_us("0.015", 15_000), 15_000)
        self.assertEqual(duration_to_us("1.005", 15_000), MAX_DURATION_US)
        for invalid in ("0", "-0.015", "NaN", "Infinity", "0.016", "1.02"):
            with self.subTest(invalid=invalid), self.assertRaises(BenchmarkError):
                duration_to_us(invalid, 15_000)

    def test_command_always_selects_exactly_one_fly(self) -> None:
        command = build_command(
            upstream_fixture(),
            Path("/data"),
            15_000,
            1,
            Path("/runs"),
        )
        index = command.index("--flies")
        self.assertEqual(command[index + 1], "1")
        self.assertEqual(command[command.index("--duration-s") + 1], "0.015")
        self.assertNotIn("--allow-dirty-tree", command)

    def test_summary_requires_one_fly_and_non_evidence_status(self) -> None:
        upstream = upstream_fixture()
        summary = {
            "batch": {"size": 1, "labels": ["fly-01"]},
            "code_commit": upstream.commit,
            "duration_us": 15_000,
            "intervals": 1,
            "evidence_grade": False,
            "validation_tier_awarded": None,
        }
        validate_upstream_summary(summary, upstream, 15_000)
        summary["batch"]["size"] = 2
        with self.assertRaises(BenchmarkError):
            validate_upstream_summary(summary, upstream, 15_000)

    def test_summary_rejects_other_revision_or_biological_claim(self) -> None:
        upstream = upstream_fixture()
        summary = {
            "batch": {"size": 1, "labels": ["fly-01"]},
            "code_commit": "e" * 40,
            "duration_us": 15_000,
            "intervals": 1,
            "evidence_grade": False,
            "validation_tier_awarded": None,
        }
        with self.assertRaises(BenchmarkError):
            validate_upstream_summary(summary, upstream, 15_000)
        summary["code_commit"] = upstream.commit
        summary["evidence_grade"] = True
        with self.assertRaises(BenchmarkError):
            validate_upstream_summary(summary, upstream, 15_000)

    def test_data_preflight_reports_missing_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            data_root = Path(directory)
            artifact = data_root / "evidence/demo01/demo01-visual-operating-point-v1.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(json.dumps({"selected": None}), encoding="utf-8")
            missing = missing_data_inputs(data_root)
        self.assertIn("imported MaleCNS graph", missing)
        self.assertIn("selected value in frozen DEMO-01 operating-point artifact", missing)

    def test_report_metadata_keeps_single_fly_and_scientific_boundary(self) -> None:
        report = _base_report(
            output=Path("run.json"),
            upstream=upstream_fixture(),
            duration_us=15_000,
            seed=1,
            status="not_run",
        )
        self.assertEqual(report["simulated_fly_count"], 1)
        self.assertFalse(report["evidence_grade"])
        self.assertIsNone(report["validation_tier_awarded"])
        self.assertIn("process_wall_seconds", report["metrics"])
        self.assertIn("gpu", report["telemetry"])
        self.assertIn("system_memory", report["telemetry"])
        self.assertIn("native_engine", report["source"])
        self.assertIn("nvcc_version_output", report["runtime"])
        self.assertNotIn("/home/", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
