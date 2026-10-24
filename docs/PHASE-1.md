# Phase 1: one-fly foundation and benchmark

## Selected simulator and provenance

The initial candidate is [Fly.exe](https://github.com/Ibtisam-Mohammad/Fly.exe), whose pinned checkout calls itself **MaleCNS Virtual Fly**. This project uses it as a separate source checkout and invokes its existing runner; Fly Heaven does not copy upstream implementation files or datasets into its Git history.

| Component | Pin | Role and terms |
|---|---|---|
| Fly.exe / MaleCNS Virtual Fly | Commit `cc25411e00f945e7330cb6f400372cce315c0802` | Existing full-connectome neural simulation with an embodied FlyGym/MuJoCo route; GPL-2.0-or-later. The source describes itself as pre-alpha. |
| Python / uv | Python `>=3.12,<3.15`; uv `0.12.5` | The simulator's declared Python range and bootstrap version. |
| Python environment | Upstream `uv.lock`, SHA-256 `c7a02d67857dffb479a7d19c000854a8e656af32c1e709889375c773efb3b5fc` | Locked transitives include NumPy 2.5.2, PyArrow 21.0.0, FlyGym 2.1.0 and MuJoCo 3.9.0. FlyGym is source tag commit `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`; both FlyGym and MuJoCo are Apache-2.0. |
| GeNN / PyGeNN | GeNN 5.4.0, source commit `dd258075263c4b2bcb6607d230add658bcc23127` | Native CUDA build required by the selected runner; the upstream installer fetches this tag. GeNN is LGPL-2.1. CUDA Toolkit version is not pinned by Fly.exe and must be recorded from the local WSL installation. |
| Connectome | MaleCNS `v1.0`, `male-cns:v1.0` | Official download page states CC-BY. Attribute MaleCNS and preserve the exact release in any result. The `starter` input profile is downloaded separately and is not redistributed here. |

The `starter` profile includes roughly 1.9 GB of published raw tables (annotations, neurotransmitters, body statistics and aggregate connection weights). The derived graph, compiled code, Python environment and run output need additional free space. Check the chosen WSL filesystem with `df -h "$FLYSIM_DATA_ROOT"` before downloading; the project does not impose a disk-size guess.

Primary sources: [pinned Fly.exe source](https://github.com/Ibtisam-Mohammad/Fly.exe/tree/cc25411e00f945e7330cb6f400372cce315c0802), [upstream license](https://github.com/Ibtisam-Mohammad/Fly.exe/blob/cc25411e00f945e7330cb6f400372cce315c0802/LICENSE), [MaleCNS download and license](https://male-cns.janelia.org/download/), [FlyGym v2.1.0 source](https://github.com/NeLy-EPFL/flygym/tree/ca65a510c2afe6ac61c51df4f274c8d190c2f95f) and [license](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/LICENSE), [GeNN 5.4.0 source](https://github.com/genn-team/genn/tree/dd258075263c4b2bcb6607d230add658bcc23127) and [license](https://github.com/genn-team/genn/blob/dd258075263c4b2bcb6607d230add658bcc23127/LICENSE), and [NVIDIA's CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

Static API review matched the code used by Fly.exe to the pinned FlyGym release: `NeuroMechFly`, `FlatGroundWorld` and `Simulation` are exported by FlyGym 2.1.0; the tagged implementation provides `add_joints`, `add_actuators`, `add_leg_adhesion`, `Simulation.step`, `get_joint_angles` and `set_actuator_inputs`. Fly.exe imports the published `HybridTurningController` from its 2.1 demo module. This is source inspection, not a runtime compatibility test on CUDA/WSL.

## What the one-fly route runs

`tools/benchmark_one_fly.py` checks the upstream commit, lock and runner interface before use. Its `run` command always supplies `--flies 1`, selects the upstream `exact` variant, and validates the resulting `summary.json` reports batch size one, the expected duration and commit, `evidence_grade: false`, and no validation tier. The upstream scenario currently declares twelve flies; its checked runner truncates the scenario to the first requested fly before constructing the world and neural batch.

The upstream command requires an imported MaleCNS graph, annotations, transmitter predictions and a selected DEMO-01 operating-point artifact. It refuses to invent its own operating point. If the artifact is missing or has no selection, the wrapper writes a `not_run` report naming that blocker. Producing the artifact with the upstream `scripts/run_demo01_visual_search.py` is a separate, finite but potentially long calibration step: its committed contract describes 108 candidates and 5.25 biological seconds per candidate. This foundation does not shorten that search or manufacture a candidate.

The default benchmark duration is 1.005 biological seconds (67 upstream 15 ms coupling intervals); the hard maximum is also 1.005 seconds. A one-interval smoke run is `0.015` seconds. The process has a 30-minute wall-clock timeout to bound compilation and execution. The report separates total subprocess wall time from upstream build and simulation time, then records simulated biological time and the upstream simulation real-time factor.

A successful run is an engineering measurement, not validation of fly behavior. The pinned upstream summary explicitly labels the route non-evidence-grade. Its walking gait is engineered, its visual scene encoder consumes object vectors rather than camera pixels, and the simulated MaleCNS specimen is paired with a FlyGym body based on a female fly model. The result is not a conscious fly or a faithful digital twin.

## Reproduce in WSL2

Run these commands manually on the target Windows PC in WSL2. They are not run by this repository's checks. Keep the upstream checkout, dataset and Fly Heaven checkout in separate directories.

```bash
export FLYHEAVEN_DIR="$HOME/src/Fly-heaven"
export FLYEXE_DIR="$HOME/src/Fly.exe"
export FLYSIM_DATA_ROOT="$HOME/fly-heaven-data"

git clone --branch codex/FLY-001-simulator-foundation \
  https://github.com/mylo123ai-hue/Fly-heaven.git "$FLYHEAVEN_DIR"
git clone https://github.com/Ibtisam-Mohammad/Fly.exe.git "$FLYEXE_DIR"
git -C "$FLYEXE_DIR" checkout --detach cc25411e00f945e7330cb6f400372cce315c0802
git -C "$FLYEXE_DIR" status --short

mkdir -p "$FLYSIM_DATA_ROOT/envs"
python3.12 -m venv "$FLYSIM_DATA_ROOT/envs/production"
"$FLYSIM_DATA_ROOT/envs/production/bin/python" -m pip install "uv==0.12.5"
source "$FLYSIM_DATA_ROOT/envs/production/bin/activate"
export UV_LINK_MODE=copy
cd "$FLYEXE_DIR"
uv sync --active --frozen --extra data --extra body --extra reference --extra render
```

Install the CUDA Toolkit for WSL2 according to NVIDIA's WSL guide, then confirm `nvidia-smi -L` and `nvcc --version` work. NVIDIA says the GPU driver is supplied by the Windows host; do not install a Linux display driver in WSL. The Fly.exe GeNN installer detects `nvcc` or uses `CUDA_PATH` and builds GeNN 5.4.0:

```bash
./scripts/install_genn.sh
flysim data sync --profile starter
flysim data validate --profile starter
flysim data import-aggregate
PYTHONPATH=src python scripts/run_demo01_visual_search.py \
  --root "$FLYSIM_DATA_ROOT" --seed 1 --progress
```

Proceed only if that search writes a non-null `selected` value to:
`$FLYSIM_DATA_ROOT/evidence/demo01/demo01-visual-operating-point-v1.json`.
The search can be long and may select no point; in that case stop and keep the failure report rather than supplying hand-picked parameters.

First run the no-simulation source/interface check:

```bash
"$FLYSIM_DATA_ROOT/envs/production/bin/python" \
  "$FLYHEAVEN_DIR/tools/benchmark_one_fly.py" verify \
  --upstream-repo "$FLYEXE_DIR"
```

Then run a one-interval smoke check and three bounded benchmark runs. Each output path must be new. The JSON report and logs remain under the selected data root; the upstream simulator also writes its full run summary there.

```bash
"$FLYSIM_DATA_ROOT/envs/production/bin/python" \
  "$FLYHEAVEN_DIR/tools/benchmark_one_fly.py" run \
  --upstream-repo "$FLYEXE_DIR" --data-root "$FLYSIM_DATA_ROOT" \
  --duration-s 0.015 --seed 1 \
  --output "$FLYSIM_DATA_ROOT/runs/FLY-001/smoke-01.json"

for run in 1 2 3; do
  "$FLYSIM_DATA_ROOT/envs/production/bin/python" \
    "$FLYHEAVEN_DIR/tools/benchmark_one_fly.py" run \
    --upstream-repo "$FLYEXE_DIR" --data-root "$FLYSIM_DATA_ROOT" \
    --duration-s 1.005 --seed 1 \
    --output "$FLYSIM_DATA_ROOT/runs/FLY-001/rtx3070-$run.json"
done
```

The wrapper samples `nvidia-smi` device-wide used/total memory and `/proc/meminfo`; these readings include unrelated GPU/system activity. It also samples the simulator process's RSS. Polling can miss brief peaks. Reports record device identity, driver, Python/PyGeNN versions, `nvcc` output when available, WSL kernel, config hashes, source commit, process status, wall and simulated times, build time, and real-time factor. Missing telemetry is `null`/unavailable with a reason; it is never reported as zero.

## Known unknowns and recovery

- The RTX 3070 8 GB WSL2 run has not been performed. Compatibility of the selected GeNN 5.4.0 native build, local CUDA Toolkit and this exact card remains for the owner-run procedure to establish.
- The upstream project is pre-alpha, and the prerequisite 108-candidate operating-point search can take substantial time or return no selected point. Without a selected point, there is no valid run to benchmark; do not create one manually.
- The pinned run is an engineering sensorimotor demo with the stated simplifications. It does not provide an autonomous VNC motor pathway, consciousness evidence, or biological validation.
- Recovery after an upstream/API mismatch is to inspect a new upstream revision and issue a separately authorized handoff. Do not silently advance the pinned commit or change the data release.
- Rollback: revert the FLY-001 task branch or close its unmerged PR. External upstream checkouts, downloaded datasets, compiled artifacts and local benchmark outputs are outside Git and can be removed independently by their owner.
