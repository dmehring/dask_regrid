# AGENTS.md

## Project Summary + Goals

`dask_regrid` provides 2D spatial regridding utilities for xarray data (`regrid_2d.py`) and supporting scripts for:

- generating synthetic/Zarr datasets,
- benchmarking serial vs parallel execution with Dask,
- validating backend differences (`xarray` vs `xesmf`),
- running scientific-correctness tests for radio-astro-like image data.

Current goals in active workflows:

- keep numerical correctness constraints stable (tests in `tests/`),
- keep parallel performance behavior understood and reproducible (benchmark scripts),
- support both `Jy/pixel` and `Jy/beam` semantics with explicit test expectations.

---

## Quickstart (Env + Install)

Assume Python `>=3.9`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e .[dev]
```

If you need XRADIO fixture generation/plotting utilities:

```bash
python -m pip install toolviper astropy s3fs casaconfig casatools matplotlib
```

If Zarr writes stall or fail with v3, prefer v2 runtime:

```bash
python -m pip install "zarr<3"
```

---

## Golden Commands

### Tests

Run all default correctness tests (xarray path; xesmf tests are gated and skip by default):

```bash
python -m pytest -q tests
```

Run selected modules:

```bash
python -m pytest -q \
  tests/test_point_source_jy_per_beam_correctness.py \
  tests/test_point_source_jy_per_pixel_correctness.py \
  tests/test_extended_gaussian_jy_per_pix_correctness.py \
  tests/test_extended_gaussian_jy_per_beam_correctness.py
```

Run opt-in xesmf tests:

```bash
RUN_XESMF_TESTS=1 python -m pytest -q tests -m xesmf
```

Expected pattern:

- default run: xarray tests pass, xesmf-marked tests skip
- `RUN_XESMF_TESTS=1`: xesmf-marked tests execute (requires working ESMF/xesmf runtime)

### Benchmark / Performance

Project benchmark script:

```bash
python benchmark_regrid.py --help
```

Recent workflow for large XRADIO-backed benchmark image generation:

```bash
python tests/util/generate_xradio_test_images.py \
  --output-dir /tmp/xradio_perf_big \
  --n-l 1024 --n-m 1024 --n-chan 256 --n-pol 1 --n-time 1 \
  --cases extended_gaussian_jy_per_pixel \
  --overwrite
```

Direct serial-vs-parallel timing pattern used:

```bash
python - <<'PY'
import time, numpy as np, xarray as xr, dask
from regrid_2d import regrid_2d_planes

da = xr.open_zarr('/tmp/xradio_perf_big/extended_gaussian_jy_per_pixel.zarr')['SKY'].isel(time=0, polarization=0).chunk({'frequency':1,'l':1024,'m':1024})
new_l = np.linspace(float(da['l'].min()), float(da['l'].max()), 640)
new_m = np.linspace(float(da['m'].min()), float(da['m'].max()), 640)

def run(scheduler, workers=None):
    obj = regrid_2d_planes(da, 'l', 'm', new_l, new_m, regridder_name='xarray', method='linear')
    cfg={'scheduler':scheduler}
    if workers is not None: cfg['num_workers']=workers
    t0=time.perf_counter()
    with dask.config.set(cfg): obj.compute()
    return time.perf_counter()-t0

print('sync', run('sync'))
print('threads8', run('threads', 8))
PY
```

Recent observed outcome (reference only): on large `1024x1024` images, `threads(8)` outperformed `sync`; speedups increased when frequency planes increased (`256 -> 512`).

### Lint / Format / Typecheck

No repository-standard lint/format/typecheck tooling is currently configured in root config files.

Assumption for contributors:

- do not introduce new mandatory tooling unless requested,
- keep style consistent with existing code and pass tests.

---

## Repo Layout (Key Folders/Files)

- `regrid_2d.py`: core regridding API (`xarray` and `xesmf` backends)
- `run_correctness_checks.py`: CLI correctness metrics/provenance report
- `benchmark_regrid.py`: scheduler/backend benchmark CLI
- `validate_regridders.py`: backend output comparison utility
- `generate_zarr_data.py`: synthetic large-array generator
- `tests/`: scientific correctness tests
- `tests/util/generate_xradio_test_images.py`: XRADIO fixture generator
- `tests/util/plot_xradio_image.py`: quick visualizer for Zarr image slices
- `regridding_session_summary.md`: running session log with benchmark/correctness notes

---

## Coding Conventions

- Prefer explicit, typed function signatures where practical.
- Keep behavior deterministic in tests (fixed fixture paths, explicit thresholds).
- For backend-specific behavior:
  - use public function params (`dim_a`, `dim_b`),
  - avoid hardcoded spatial names at call sites.
- Add actionable assertion messages in tests.
- Keep comments concise and focused on scientific/algorithmic intent.

Error handling expectations:

- fail fast on missing required metadata (for example beam metadata in `Jy/beam` workflows),
- skip tests explicitly when optional runtime dependencies are not enabled.

---

## Performance Notes / Regression Expectations

- The default performance baseline is `sync` vs `threads` on Dask-backed arrays.
- Parallelism scales with task count (notably frequency-plane chunking).
- For large XRADIO-like workloads tested recently:
  - `threads(8)` was near-optimal,
  - `threads(16)` could regress vs `threads(8)`.
- Avoid changes that significantly reduce observed thread speedup on large workloads without justification.

When changing compute paths, include:

- timings for `sync` and at least `threads(2/4/8)`,
- shape/chunk details,
- backend and method used.

---

## Numerical Constraints (Must Not Regress)

Based on current tests:

- Point source `Jy/beam`:
  - identity: preserve peak and sum to tight tolerance,
  - resample: non-negativity and centroid bounds,
  - round-trip: bounded peak degradation and centroid drift.
- Point source `Jy/pixel`:
  - identity: preserve area-weighted integrated flux,
  - resample/round-trip: bounded integrated flux ratio and centroid drift.
- Extended Gaussian (`Jy/pixel`):
  - strong integrated-flux stability expectations (`~0.2%` rel in current tests),
  - bounded peak attenuation and RMS residual.
- Extended Gaussian (`Jy/beam`):
  - emphasize peak/centroid/RMS behavior and beam metadata presence.

Any tolerance changes require rationale in PR notes.

---

## PR Expectations

Include in PR description:

1. What changed and why.
2. Exact commands run.
3. Relevant output snippets (pass/fail + benchmark numbers).
4. Any changed thresholds and justification.

Minimum evidence template:

```text
Tests:
- python -m pytest -q tests/...
- Result: <summary>

Benchmarks (if compute path changed):
- command: <exact>
- shape/chunks: <exact>
- sync: <time>
- threads(2/4/8): <times>
```

---

## Do / Don’t Rules

### Do

- Do keep imports pointing to `regrid_2d.py` (not legacy names).
- Do run module-specific tests before committing touched areas.
- Do keep xesmf tests opt-in via `RUN_XESMF_TESTS=1`.
- Do preserve existing test semantics for `Jy/pixel` vs `Jy/beam`.

### Don’t

- Don’t commit generated artifacts or local datasets by default.
  - Specifically avoid committing local fixture/output directories such as `xradio_test_images/`.
- Don’t commit external cloned repos used only for local exploration (for example local `xradio/` clone).
- Don’t hardcode personal paths, tokens, or environment-specific secrets.
- Don’t silently relax correctness tolerances without documenting why.

---

## Assumptions

1. Root repository currently has no authoritative `README.md` despite `pyproject.toml` referencing one.
2. No root CI workflow or Makefile is currently used as the canonical project runner.
3. `xesmf` execution may require less-restricted runtime permissions depending on environment (MPI/UCX behavior).
4. Commands above are intended to run from repository root on a clean checkout with dependencies installed.
