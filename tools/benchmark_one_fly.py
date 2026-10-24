#!/usr/bin/env python3
"""Verify the pinned Fly.exe interface or run its bounded, one-fly benchmark.

This adapter invokes the upstream CLI in a separate process. It does not copy or import
upstream simulator code. The GPU and memory readings are device/system-wide samples and
can include other activity on the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/Ibtisam-Mohammad/Fly.exe"
UPSTREAM_REVISION = "cc25411e00f945e7330cb6f400372cce315c0802"
EXPECTED_PYTHON = ">=3.12,<3.15"
EXPECTED_FLYGYM = "2.1.0"
EXPECTED_MUJOCO = "3.9.0"
EXPECTED_GENN = "5.4.0"
DEFAULT_DURATION = Decimal("1.005")
MAX_DURATION_US = 1_005_000
RUN_TIMEOUT_SECONDS = 30 * 60
SAMPLE_INTERVAL_SECONDS = 0.25


class BenchmarkError(ValueError):
    """A checked setup or run contract was not met."""


@dataclass(frozen=True)
class Upstream:
    root: Path
    commit: str
    scenario_path: Path
    contract_path: Path
    runner_path: Path
    coupling_us: int
    scenario_sha256: str
    contract_sha256: str
    uv_lock_sha256: str
    scenario_fly_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        raise BenchmarkError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def inspect_upstream(repository: Path, *, require_clean: bool = True) -> Upstream:
    root = repository.expanduser().resolve()
    if not root.is_dir():
        raise BenchmarkError(f"upstream repository directory does not exist: {root}")

    commit = _git(root, "rev-parse", "HEAD")
    if commit != UPSTREAM_REVISION:
        raise BenchmarkError(
            f"Fly.exe revision must be {UPSTREAM_REVISION}; found {commit}"
        )
    if require_clean and _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise BenchmarkError("Fly.exe working tree must be clean for a reproducible run")

    relative_files = {
        "scenario": Path("configs/scenarios/swarm3d-showcase.json"),
        "contract": Path("configs/experiments/demo01-visual-operating-point-v1.json"),
        "runner": Path("scripts/run_swarm3d_showcase.py"),
        "engine": Path("src/flysim/swarm3d_run.py"),
        "project": Path("pyproject.toml"),
        "lock": Path("uv.lock"),
        "genn_installer": Path("scripts/install_genn.sh"),
    }
    missing = [str(path) for path in relative_files.values() if not (root / path).is_file()]
    if missing:
        raise BenchmarkError(
            "pinned Fly.exe checkout is incomplete; missing: " + ", ".join(missing)
        )

    project = tomllib.loads((root / relative_files["project"]).read_text(encoding="utf-8"))
    project_info = project["project"]
    body_extra = project_info["optional-dependencies"]["body"]
    native = project["tool"]["flysim"]["native-dependencies"]["pygenn"]
    if project_info["requires-python"] != EXPECTED_PYTHON:
        raise BenchmarkError("pinned Fly.exe Python requirement changed")
    if f"flygym=={EXPECTED_FLYGYM}" not in body_extra:
        raise BenchmarkError("pinned FlyGym version is missing from Fly.exe")
    if not any(item.startswith("mujoco>=3.9,<3.10") for item in body_extra):
        raise BenchmarkError("pinned MuJoCo version range is missing from Fly.exe")
    if not native.endswith(f"/tags/{EXPECTED_GENN}.zip"):
        raise BenchmarkError("pinned GeNN source version is missing from Fly.exe")

    lock = tomllib.loads((root / relative_files["lock"]).read_text(encoding="utf-8"))
    locked = {item["name"]: item["version"] for item in lock["package"]}
    if locked.get("flygym") != EXPECTED_FLYGYM or locked.get("mujoco") != EXPECTED_MUJOCO:
        raise BenchmarkError("FlyGym or MuJoCo version in uv.lock does not match the pin")

    scenario_path = root / relative_files["scenario"]
    contract_path = root / relative_files["contract"]
    runner_path = root / relative_files["runner"]
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    runner_source = runner_path.read_text(encoding="utf-8")
    engine_source = (root / relative_files["engine"]).read_text(encoding="utf-8")
    if not isinstance(scenario.get("flies"), list) or not scenario["flies"]:
        raise BenchmarkError("pinned scenario does not contain a fly to select")
    if int(contract.get("coupling_us", 0)) <= 0:
        raise BenchmarkError("pinned benchmark contract has no positive coupling interval")
    for option in ("--flies", "--duration-s", "--scenario", "--contract", "--output-root"):
        if not re.search(rf"add_argument\(\s*['\"]{re.escape(option)}['\"]", runner_source):
            raise BenchmarkError(f"pinned upstream runner no longer supports {option}")
    if "scenario.flies[: args.flies]" not in runner_source:
        raise BenchmarkError(
            "pinned upstream runner no longer truncates to the requested fly count"
        )
    if "batch_size=fly_count" not in engine_source:
        raise BenchmarkError("pinned upstream engine no longer batches exactly scenario.flies")

    return Upstream(
        root=root,
        commit=commit,
        scenario_path=scenario_path,
        contract_path=contract_path,
        runner_path=runner_path,
        coupling_us=int(contract["coupling_us"]),
        scenario_sha256=sha256_file(scenario_path),
        contract_sha256=sha256_file(contract_path),
        uv_lock_sha256=sha256_file(root / relative_files["lock"]),
        scenario_fly_count=len(scenario["flies"]),
    )


def duration_to_us(value: str | Decimal, coupling_us: int) -> int:
    try:
        seconds = Decimal(str(value))
    except InvalidOperation as exc:
        raise BenchmarkError("duration must be a decimal number of seconds") from exc
    if not seconds.is_finite() or seconds <= 0:
        raise BenchmarkError("duration must be finite and greater than zero")
    micros = seconds * Decimal(1_000_000)
    if micros != micros.to_integral_value():
        raise BenchmarkError("duration must resolve to an integer number of microseconds")
    duration_us = int(micros)
    if duration_us > MAX_DURATION_US:
        raise BenchmarkError(f"duration may not exceed {MAX_DURATION_US / 1_000_000:g} seconds")
    if duration_us % coupling_us:
        raise BenchmarkError(
            f"duration must be a whole {coupling_us}-microsecond coupling interval"
        )
    return duration_us


def _duration_string(duration_us: int) -> str:
    return f"{Decimal(duration_us) / Decimal(1_000_000):f}".rstrip("0").rstrip(".")


def build_command(
    upstream: Upstream,
    data_root: Path,
    duration_us: int,
    seed: int,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(upstream.runner_path),
        "--root",
        str(data_root),
        "--contract",
        str(upstream.contract_path),
        "--scenario",
        str(upstream.scenario_path),
        "--duration-s",
        _duration_string(duration_us),
        "--seed",
        str(seed),
        "--variant",
        "exact",
        "--flies",
        "1",
        "--output-root",
        str(output_root),
    ]


def missing_data_inputs(data_root: Path) -> list[str]:
    required = {
        "imported MaleCNS graph": data_root / "derived/male-cns-v1.0/graph",
        "MaleCNS body annotations": data_root
        / "raw/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "MaleCNS neurotransmitter predictions": data_root
        / "raw/male-cns-v1.0/body-neurotransmitters-male-cns-v1.0.feather",
        "frozen DEMO-01 operating-point artifact": data_root
        / "evidence/demo01/demo01-visual-operating-point-v1.json",
    }
    missing = [
        label
        for label, path in required.items()
        if not (path.is_dir() if label == "imported MaleCNS graph" else path.is_file())
    ]
    operating = required["frozen DEMO-01 operating-point artifact"]
    if operating.is_file():
        try:
            if not json.loads(operating.read_text(encoding="utf-8")).get("selected"):
                missing.append("selected value in frozen DEMO-01 operating-point artifact")
        except (OSError, json.JSONDecodeError):
            missing.append("readable frozen DEMO-01 operating-point artifact")
    return missing


def _read_gpu(index: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name,memory.used,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi returned a nonzero status")
    line = next((item for item in result.stdout.splitlines() if item.strip()), "")
    fields = [item.strip() for item in line.split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi response: {line!r}")
    return {
        "name": fields[0],
        "used_mib": int(fields[1]),
        "total_mib": int(fields[2]),
        "driver_version": fields[3],
    }


def _read_system_memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, separator, rest = line.partition(":")
        if separator and key in {"MemTotal", "MemAvailable"}:
            values[key] = int(rest.split()[0])
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("/proc/meminfo is missing MemTotal or MemAvailable")
    return values["MemTotal"] // 1024, (values["MemTotal"] - values["MemAvailable"]) // 1024


def _read_process_rss_mib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


class TelemetrySampler:
    def __init__(self, gpu_index: int, interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pid: int | None = None
        self._gpu_samples: list[dict[str, Any]] = []
        self._gpu_error: str | None = None
        self._system_samples: list[tuple[int, int]] = []
        self._process_rss_samples: list[int] = []

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._poll, name="flybench-telemetry", daemon=True)
        self._thread.start()

    def set_pid(self, pid: int) -> None:
        self._pid = pid

    def _sample(self) -> None:
        try:
            self._system_samples.append(_read_system_memory())
        except (OSError, ValueError, RuntimeError):
            pass
        if self._pid is not None:
            rss = _read_process_rss_mib(self._pid)
            if rss is not None:
                self._process_rss_samples.append(rss)
        if self._gpu_error is None:
            try:
                self._gpu_samples.append(_read_gpu(self.gpu_index))
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                self._gpu_error = str(exc)

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
        self._sample()
        gpu: dict[str, Any]
        if self._gpu_samples:
            used = [sample["used_mib"] for sample in self._gpu_samples]
            first = self._gpu_samples[0]
            gpu = {
                "available": True,
                "name": first["name"],
                "driver_version": first["driver_version"],
                "device_total_mib": first["total_mib"],
                "baseline_device_used_mib": used[0],
                "peak_device_used_mib": max(used),
                "peak_delta_from_baseline_mib": max(used) - used[0],
                "sample_count": len(used),
                "measurement": "nvidia-smi device-wide polling; other GPU processes are included",
            }
        else:
            gpu = {
                "available": False,
                "reason": self._gpu_error or "no successful nvidia-smi sample",
                "sample_count": 0,
            }
        if self._system_samples:
            totals = [sample[0] for sample in self._system_samples]
            used = [sample[1] for sample in self._system_samples]
            system: dict[str, Any] = {
                "available": True,
                "total_mib": max(totals),
                "baseline_used_mib": used[0],
                "peak_used_mib": max(used),
                "peak_delta_from_baseline_mib": max(used) - used[0],
                "sample_count": len(used),
                "measurement": "system-wide MemTotal minus MemAvailable from /proc/meminfo",
            }
        else:
            system = {"available": False, "reason": "/proc/meminfo was unavailable"}
        process: dict[str, Any] = {
            "available": bool(self._process_rss_samples),
            "peak_rss_mib": max(self._process_rss_samples) if self._process_rss_samples else None,
            "sample_count": len(self._process_rss_samples),
            "measurement": "simulator child process VmRSS from /proc/<pid>/status",
        }
        return {"gpu": gpu, "system_memory": system, "process_memory": process}


def _read_git_metadata(repository: Path) -> dict[str, Any]:
    try:
        return {
            "commit": _git(repository, "rev-parse", "HEAD"),
            "working_tree_dirty": bool(
                _git(repository, "status", "--porcelain", "--untracked-files=all")
            ),
        }
    except (BenchmarkError, OSError):
        return {"commit": None, "working_tree_dirty": None}


def _read_runtime_metadata() -> tuple[dict[str, Any], str | None]:
    pygenn = subprocess.run(
        [sys.executable, "-c", "import pygenn; print(pygenn.__version__)"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    pygenn_version = pygenn.stdout.strip().splitlines()[-1] if pygenn.stdout.strip() else None
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "wsl_kernel": platform.release(),
        "pygenn_version": pygenn_version,
        "nvcc_version_output": None,
    }
    blocker = None
    if pygenn.returncode:
        blocker = "PyGeNN import failed: " + (pygenn.stderr.strip() or "no error text")[:500]
    elif pygenn_version != EXPECTED_GENN:
        blocker = f"PyGeNN version must be {EXPECTED_GENN}; found {pygenn_version!r}"

    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        runtime["nvcc_version_output"] = (
            nvcc.stdout.strip() if nvcc.returncode == 0 else None
        )
        if nvcc.returncode:
            runtime["nvcc_unavailable_reason"] = (
                nvcc.stderr.strip() or f"nvcc exited with status {nvcc.returncode}"
            )[:500]
    except (OSError, subprocess.TimeoutExpired) as exc:
        runtime["nvcc_unavailable_reason"] = str(exc)[:500]
    return runtime, blocker


def _signal_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def validate_upstream_summary(
    summary: dict[str, Any], upstream: Upstream, duration_us: int
) -> None:
    batch = summary.get("batch", {})
    if batch.get("size") != 1 or len(batch.get("labels", [])) != 1:
        raise BenchmarkError("upstream summary does not confirm exactly one simulated fly")
    if summary.get("code_commit") != upstream.commit:
        raise BenchmarkError("upstream run summary records a different source commit")
    if summary.get("duration_us") != duration_us:
        raise BenchmarkError("upstream run summary records a different biological duration")
    if summary.get("intervals") != duration_us // upstream.coupling_us:
        raise BenchmarkError("upstream run summary records an unexpected interval count")
    if summary.get("evidence_grade") is not False:
        raise BenchmarkError("upstream summary must explicitly remain non-evidence-grade")
    if summary.get("validation_tier_awarded") is not None:
        raise BenchmarkError("upstream summary unexpectedly reports a validation tier")


def _base_report(
    *,
    output: Path,
    upstream: Upstream | None,
    duration_us: int | None,
    seed: int,
    status: str,
    blocker: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task": "FLY-001",
        "status": status,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "simulated_fly_count": 1,
        "run_parameters": {
            "variant": "exact",
            "seed": seed,
            "requested_biological_duration_us": duration_us,
            "maximum_biological_duration_us": MAX_DURATION_US,
            "gpu_index": None,
        },
        "source": {
            "simulator": UPSTREAM_REPOSITORY,
            "simulator_commit": upstream.commit if upstream else None,
            "simulator_uv_lock_sha256": upstream.uv_lock_sha256 if upstream else None,
            "python": platform.python_version(),
            "native_engine": f"GeNN/PyGeNN {EXPECTED_GENN}",
            "harness_commit": _read_git_metadata(Path(__file__).resolve().parents[1]),
        },
        "model_and_config": {
            "dataset": "MaleCNS v1.0 (CC-BY; attribution required)",
            "scenario_sha256": upstream.scenario_sha256 if upstream else None,
            "experiment_contract_sha256": upstream.contract_sha256 if upstream else None,
            "operating_point_sha256": None,
        },
        "metrics": {
            "process_wall_seconds": None,
            "upstream_build_seconds": None,
            "upstream_simulation_seconds": None,
            "simulated_biological_seconds": None,
            "simulation_real_time_factor": None,
        },
        "telemetry": {
            "gpu": {"available": False, "reason": "simulator was not started"},
            "system_memory": {"available": False, "reason": "simulator was not started"},
            "process_memory": {"available": False, "reason": "simulator was not started"},
        },
        "runtime": {"pygenn_version": None, "nvcc_version_output": None},
        "evidence_grade": False,
        "validation_tier_awarded": None,
        "claim_boundary": (
            "A successful run is an engineering simulation measurement, not evidence of "
            "consciousness or biologically validated behavior. The upstream route uses an "
            "engineered gait and an object-vector visual encoding."
        ),
        "blocker": blocker,
        "upstream_summary": None,
        "stdout_log": None,
        "stderr_log": None,
        "report_file": output.name,
    }


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def run_benchmark(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    upstream_path = args.upstream_repo.expanduser().resolve()
    if output.exists():
        raise BenchmarkError(f"refusing to overwrite existing report: {output}")
    if output.is_relative_to(upstream_path):
        raise BenchmarkError("benchmark report must be outside the upstream checkout")
    try:
        upstream = inspect_upstream(args.upstream_repo)
        duration_us = duration_to_us(args.duration_s, upstream.coupling_us)
        if data_root.is_relative_to(upstream.root):
            raise BenchmarkError("dataset and run data must be outside the upstream checkout")
    except (BenchmarkError, OSError) as exc:
        report = _base_report(
            output=output,
            upstream=None,
            duration_us=None,
            seed=args.seed,
            status="not_run",
            blocker=str(exc),
        )
        _write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    report = _base_report(
        output=output,
        upstream=upstream,
        duration_us=duration_us,
        seed=args.seed,
        status="not_run",
    )
    if data_root.is_relative_to(upstream.root):
        report["blocker"] = "dataset and run data must be outside the upstream checkout"
        _write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    operating_point = data_root / "evidence/demo01/demo01-visual-operating-point-v1.json"
    if operating_point.is_file():
        report["model_and_config"]["operating_point_sha256"] = sha256_file(operating_point)
    missing = missing_data_inputs(data_root)
    if missing:
        report["blocker"] = "required upstream inputs are missing: " + ", ".join(missing)
        _write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    runtime, runtime_blocker = _read_runtime_metadata()
    report["runtime"] = runtime
    if runtime_blocker is not None:
        report["blocker"] = runtime_blocker
        _write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    if not 0 <= args.seed <= 2**31 - 1:
        raise BenchmarkError("seed must be in [0, 2**31-1]")
    output_root = output.parent / f"{output.stem}.upstream-runs"
    if output_root.exists():
        raise BenchmarkError(f"refusing to reuse existing upstream output directory: {output_root}")
    if output_root.is_relative_to(upstream.root):
        raise BenchmarkError("benchmark output must be outside the upstream checkout")

    output_root.mkdir(parents=True)
    stdout_path = output.with_suffix(".stdout.log")
    stderr_path = output.with_suffix(".stderr.log")
    command = build_command(upstream, data_root, duration_us, args.seed, output_root)
    report["run_parameters"].update(
        {"gpu_index": args.gpu_index, "duration_seconds": _duration_string(duration_us)}
    )
    report["reproducible_invocation"] = {
        "upstream_script": "scripts/run_swarm3d_showcase.py",
        "variant": "exact",
        "fly_count": 1,
        "duration_seconds": _duration_string(duration_us),
        "seed": args.seed,
        "upstream_revision": upstream.commit,
    }
    env = os.environ.copy()
    env["FLYSIM_DATA_ROOT"] = str(data_root)
    env.setdefault("PYTHONUNBUFFERED", "1")
    sampler = TelemetrySampler(args.gpu_index)
    sampler.start()
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    interrupted = False
    launch_error: str | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=upstream.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        sampler.set_pid(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=RUN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_process_group(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_process_group(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
        except KeyboardInterrupt:
            interrupted = True
            _signal_process_group(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_process_group(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
    except OSError as exc:
        launch_error = str(exc)
    finally:
        process_wall_seconds = time.perf_counter() - started
        telemetry = sampler.finish()

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    report["stdout_log"] = stdout_path.name
    report["stderr_log"] = stderr_path.name
    report["metrics"]["process_wall_seconds"] = round(process_wall_seconds, 6)
    report["telemetry"] = telemetry
    report["hardware"] = {
        "target": "NVIDIA RTX 3070 8 GB",
        "reported_device": telemetry["gpu"].get("name"),
        "reported_hardware_matches_target": (
            "rtx 3070" in telemetry["gpu"].get("name", "").lower()
            and 7500 <= telemetry["gpu"].get("device_total_mib", 0) <= 8500
        ),
    }

    summary_files = sorted(output_root.rglob("summary.json"))
    if summary_files:
        summary_path = summary_files[0]
        report["upstream_summary"] = str(summary_path.relative_to(output_root))
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_upstream_summary(summary, upstream, duration_us)
            report["metrics"].update(
                {
                    "upstream_build_seconds": summary.get("build_seconds"),
                    "upstream_simulation_seconds": summary.get("simulation_seconds"),
                    "simulated_biological_seconds": duration_us / 1_000_000,
                    "simulation_real_time_factor": summary.get("biological_per_wall"),
                }
            )
        except (BenchmarkError, OSError, json.JSONDecodeError, TypeError) as exc:
            report["blocker"] = f"upstream summary validation failed: {exc}"

    if interrupted:
        report["status"] = "interrupted"
        report["blocker"] = "run was interrupted by the operator"
        exit_code = 130
    elif timed_out:
        report["status"] = "timed_out"
        report["blocker"] = f"simulator exceeded the {RUN_TIMEOUT_SECONDS}-second process limit"
        exit_code = 124
    elif launch_error is not None:
        report["status"] = "failed"
        report["blocker"] = f"could not start upstream simulator: {launch_error}"
        exit_code = 2
    elif process is None or process.returncode != 0:
        report["status"] = "failed"
        code = None if process is None else process.returncode
        report["blocker"] = f"upstream simulator exited with status {code}"
        exit_code = 2
    elif len(summary_files) != 1 or report["blocker"] is not None:
        report["status"] = "failed"
        if report["blocker"] is None:
            report["blocker"] = f"expected one summary.json, found {len(summary_files)}"
        exit_code = 2
    else:
        report["status"] = "completed"
        report["blocker"] = None
        exit_code = 0
    _write_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


def _add_upstream_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--upstream-repo", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="check the pinned upstream revision and interface")
    _add_upstream_argument(verify)
    run = commands.add_parser("run", help="run one bounded upstream fly and capture metrics")
    _add_upstream_argument(run)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--duration-s", default=str(DEFAULT_DURATION))
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "verify":
        try:
            upstream = inspect_upstream(args.upstream_repo)
            report = {
                "status": "static_upstream_check_passed",
                "simulation_run": False,
                "simulator": UPSTREAM_REPOSITORY,
                "commit": upstream.commit,
                "scenario_declared_fly_count": upstream.scenario_fly_count,
                "single_fly_option_verified": True,
                "coupling_us": upstream.coupling_us,
                "scenario_sha256": upstream.scenario_sha256,
                "experiment_contract_sha256": upstream.contract_sha256,
                "uv_lock_sha256": upstream.uv_lock_sha256,
            }
        except (BenchmarkError, OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            print(f"upstream verification failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not 0 <= args.seed <= 2**31 - 1:
        parser.error("--seed must be in [0, 2**31-1]")
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    try:
        return run_benchmark(args)
    except (BenchmarkError, OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        print(f"benchmark setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
