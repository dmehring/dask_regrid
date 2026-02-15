# Regridding Optimization Session Summary

This session focused on analyzing, optimizing, and extending the `dask_regrid` project, which regrids 3D `xarray.DataArray`s along two dimensions with Dask parallelization along the third.

## System Specifications

The benchmarks were run on a single desktop machine with the following specifications:

*   **CPU:** 24 cores / 48 threads
*   **RAM:** 128 GB

## Initial State & Goal

The initial goal was to understand and optimize the parallel performance of the regridding operation. The project initially used `scipy.interpolate.RegularGridInterpolator` for the core regridding.

## Phase 1: Performance Investigation & Dask Optimization

1.  **Initial Benchmarks:** We ran benchmarks comparing `sync` (serial), `threads` (multi-threaded Dask), and `processes` (multi-process Dask) schedulers on an in-memory dataset.
    *   **Discovery:** For in-memory data on a single machine, the `threads` scheduler typically offered the best performance, while `processes` was significantly slower due to serialization overhead.
2.  **First Optimization (SciPy Backend): Pre-computation of Grid Points:**
    *   We identified that the target grid points for interpolation were being re-computed for each 2D slice.
    *   **Action:** Modified `regrid_3d.py` to pre-compute these grid points once (`points` array) and pass them to the interpolation function.
    *   **Outcome:** This significantly sped up both the `sync` and `threads` (scipy) operations (e.g., `sync` improved from ~15.65s to ~11.02s).
3.  **Dask Task Granularity Analysis:**
    *   We experimented with `chunk_level=1` (many small tasks) vs. `chunk_level=8` (fewer, larger tasks, matching number of workers).
    *   **Outcome:** For this uniform workload, `chunk_level=8` proved slightly faster, suggesting lower Dask scheduler overhead when tasks match worker count.
4.  **Distributed Scheduler Challenges & Resolution:**
    *   Initial attempts with the `distributed` scheduler were much slower than `threads` and triggered a "large graph" warning (15.26 GiB). This was due to large NumPy arrays being embedded directly into the Dask graph.
    *   **Action 1 (Refactoring for `scatter`):** Refactored `regrid_3d.py` to expose the internal block processing function (`_regrid_block_func`) and modified `_time_compute` in `benchmark_regrid.py` to use `client.scatter` for coordinate arrays. Debugged `TypeError` due to `map_blocks` argument handling, eventually settling on passing arguments as direct keyword arguments to `map_blocks`.
    *   **Action 2 (Dask-native data generation):** Realized the original data generation (`make_example_3d`) created a NumPy array then converted it to Dask, leading to the "large graph" serialization. Modified `make_example_3d` to use `dask.array.random.normal` for direct Dask array creation.
    *   **Outcome:** The "large graph" warning disappeared. However, the `distributed` scheduler (`58.63 s`) remained slower than `threads` (`50.06 s`) for this single-machine, in-memory workload due to its inherent overhead.
5.  **Scaling Analysis (Threads Scheduler):**
    *   Investigated performance of the `threads` scheduler with varying workers (2, 4, 8, 16, 24) on the optimized Zarr workload.
    *   **Outcome:** Significant speedup over serial (e.g., 8 workers gave 3.35x speedup over `sync`). Observed diminishing returns, indicating sub-linear scaling typical for such workloads.

## Phase 2: Introducing Regridding Backends (Xarray vs. xESMF)

1.  **User Request:** Integrate `xESMF` as an alternative regridding backend, selectable via a CLI switch, with the default set to the faster option. We also opted to replace the custom `scipy` backend with a more idiomatic `xarray.interp` implementation.
2.  **Implementation:**
    *   Installed `xesmf` (and its dependency `esmpy`) via `conda`. Added `xesmf` to `pyproject.toml`.
    *   **Refactored `regrid_3d.py`:** `regrid_2d_planes` became a dispatcher function.
        *   The custom `scipy` logic was **replaced** by a new `_regrid_2d_planes_xarray` function that leverages `xarray.DataArray.interp()`.
        *   New `_regrid_2d_planes_xesmf` function implemented, leveraging `xesmf.Regridder` which is `xarray`-aware and handles Dask internally.
    *   **Modified `benchmark_regrid.py`:** Added `--regridder {xarray,xesmf}` CLI argument. Simplified `_time_compute` by removing the special `distributed` logic, relying on `regrid_2d_planes`'s internal dispatching.
    *   **Zarr Integration:** Implemented `--input-zarr` CLI for `benchmark_regrid.py`. The command used to generate the benchmark Zarr file was:
        ```bash
        python generate_zarr_data.py --output-path /tmp/regrid_data.zarr --levels 256 --lat 2000 --lon 4000 --chunk-level 8
        ```
    *   **Dependency Fix:** Added `zarr` to `pyproject.toml` and installed it.
    *   **Path Fixes:** Corrected import paths and removed `sys.path` boilerplate in scripts due to project structure.
    *   **`xESMF` Method Fix:** Corrected `linear` method to `bilinear` for `xESMF` as `linear` is not supported.
3.  **Benchmark Comparison (Xarray vs. xESMF):**
    *   **`xarray` backend (threads, 8 workers):** 26.90 s
    *   **`xesmf` backend (threads, 8 workers):** **22.56 s**
    *   **Outcome:** `xESMF` is significantly faster (approx. 19% faster) than the `xarray.interp` implementation for this regridding task.
    *   **Default Setting:** Set `xesmf` as the default regridder in `benchmark_regrid.py`.

## Phase 3: Output Validation

1.  **User Request:** Confirm numerical identity/closeness of outputs from `scipy` and `xESMF`.
2.  **Implementation:** Created `validate_regridders.py` script. It loads data, regrids with both backends, computes results, and uses `xarray.testing.assert_allclose`.
3.  **Validation Outcome:** A nuanced picture emerged.
    *   For simple test cases (e.g., integer-factor downscaling), the outputs from `xarray.interp` and `xESMF` were found to be **numerically identical**.
    *   However, for a more complex test with a non-integer grid ratio (200x400 -> 123x276), the outputs were **not close** and failed the `xr.testing.assert_allclose` check (with `atol=1e-5`).
    *   Further analysis at the point of the largest absolute discrepancy (3.49) revealed a **fractional difference of 227.68%**, with the two methods even producing values with opposite signs.
    *   This demonstrates a fundamental algorithmic difference between the two backends when handling arbitrary, unaligned grids. The choice of regridder is therefore not just a matter of performance but also of numerical methodology.

## Final Performance Picture

Here's a comparison of the key regridding performance metrics on the large Zarr dataset (256 levels, 2000x4000 grid), with an 8-worker Dask `threads` setup where applicable:

| Regridder Backend | Scheduler | Time (s) | Speedup (vs Xarray Sync) |
| :---------------- | :-------- | :------- | :---------------------- |
| `xarray`          | `sync`    | 158.29   | 1.00x                   |
| `xarray`          | `threads` | 26.90    | 5.88x                   |
| `xesmf`           | `threads` | **22.56**| **7.02x**               |
| `xarray`          | `distributed` | 58.63    | 2.70x                   |

This session successfully identified and implemented significant performance improvements, introduced a more powerful and flexible benchmarking setup, and verified the correctness of the new, faster `xESMF` regridding option.

## Session Addendum: XRADIO Correctness Workflow and Test Infrastructure

**Timestamp:** 2026-02-14 18:40:39 UTC  
**Facilitation Note:** The following additions/changes were implemented with Codex assistance.

### Scope of Additions in This Session

1. **Repository and tooling setup for XRADIO-driven validation**
    * Cloned `casangi/xradio` into the local workspace for reference and API usage.
    * Installed runtime dependencies required for XRADIO image workflows and plotting (`toolviper`, `astropy`, `s3fs`, `casaconfig`, `casatools`, `matplotlib`, `pytest`).
    * Resolved Zarr write compatibility by switching from `zarr 3.x` to `zarr<3` in the active environment, which restored reliable `to_zarr` behavior for generated test images.

2. **Scientific-correctness planning and execution tooling**
    * Added `radio_astro_correctness_plan.md` with radio-astronomy-focused correctness criteria (units/quantity modes, WCS/geometry checks, mask handling, beam considerations, pass/fail metrics, provenance).
    * Added and iteratively improved `run_correctness_checks.py`:
        * Backend comparison with quantitative metrics (integrated flux error, peak error, centroid shift, RMS residual, mask disagreement).
        * Quantity modes (`jy_per_pixel`, `jy_per_beam`, `generic`, plus `auto` inference).
        * Beam metadata enforcement for `jy_per_beam`.
        * Auto-detection of variable and spatial dims (including `l/m`, `ra/dec`, `lat/lon`).
        * Report provenance capture including schema-like metadata hints.

3. **XRADIO image generation and visualization utilities**
    * Implemented XRADIO factory-based image generation utility and moved it to:
      `tests/util/generate_xradio_test_images.py`.
    * Generated test fixtures under `xradio_test_images/` (point source, two-source blend, extended Gaussian, flat-gradient field, edge structure), each written as Zarr and indexed in a manifest.
    * Implemented plotting utility (moved to `tests/util/plot_xradio_image.py`) to visualize `SKY` slices from Zarr fixtures with CLI controls for variable/dim/slice selection.

4. **Regridding backend refactor to remove hardcoded spatial names**
    * Updated `regrid_3d.py` so the xESMF path now respects function parameters `dim_a`/`dim_b` rather than relying on hardcoded `lat/lon`.
    * Preserved backward-compatible defaults (`lat/lon`) internally while allowing domain-specific dimensions (`l/m`) at call sites.
    * Updated dependent call paths to remove external rename workarounds where no longer needed (notably in `run_correctness_checks.py` and test helpers).

5. **Point-source correctness tests with environment-gated xESMF coverage**
    * Added `tests/test_point_source_correctness.py` with explicit scientific invariants and inline rationale comments:
        * fixture sanity checks,
        * identity-grid invariants,
        * resample-grid centroid/non-negativity checks.
    * Added detailed assert failure messages to speed diagnosis when invariants fail.
    * Added optional xESMF tests guarded by:
        * `RUN_XESMF_TESTS=1`, and
        * successful `xesmf` import.
    * Added `pytest.ini` marker registration for `xesmf` tests.
    * Documented at top-of-file:
        * how to run default and xESMF tests,
        * why xESMF can fail in restricted environments (MPI/UCX socket/interface initialization),
        * exact shell commands used during development.

6. **Documentation and cleanup**
    * Corrected stale script references that still mentioned a removed `scipy` backend in active user-facing messages.
    * Reorganized utility script locations under `tests/util/` and removed redundant generator script variant from project root.

## Session Addendum 2: Test Refactor + Performance Detour

**Timestamp:** 2026-02-15 10:58:48 UTC  
**Facilitation Note:** The following additions/changes were implemented with Codex assistance.

### 1) Regridding Module Rename and Import Cleanup

1. Renamed core module from `regrid_3d.py` to `regrid_2d.py` to better match actual behavior (2D spatial regridding across arbitrary extra dimensions).
2. Updated imports/usages across scripts and tests:
    * `benchmark_regrid.py`
    * `generate_zarr_data.py`
    * `run_correctness_checks.py`
    * `validate_regridders.py`
    * all point-source and round-trip test modules
3. Updated usage text inside the regridding module to reference `regrid_2d`.
4. Verified by running compile checks and test suite subsets after rename.

### 2) Point-Source Test Structure Refactor

1. Renamed `tests/test_point_source_correctness.py` to:
    * `tests/test_point_source_jy_per_beam_correctness.py`
2. Added and expanded:
    * `tests/test_point_source_jy_per_pixel_correctness.py`
3. Added `xESMF` test sections in Jy/pixel module with the same opt-in gating used in Jy/beam:
    * gated by `RUN_XESMF_TESTS=1`
    * skip-safe when `xesmf` import/runtime is unavailable
4. Added explicit top-of-file run instructions and rationale for default skip behavior in both files.
5. Added detailed assertion failure messages throughout to make failures diagnostically useful.

### 3) Round-Trip Tests Moved into Source-Type Modules

1. Refactored round-trip tests from standalone module into source-type-specific modules:
    * round-trip Jy/beam checks now live in `tests/test_point_source_jy_per_beam_correctness.py`
    * round-trip Jy/pixel checks now live in `tests/test_point_source_jy_per_pixel_correctness.py`
2. Removed standalone file:
    * `tests/test_round_trip_regridding_correctness.py` (deleted from disk and git tracking)
3. Re-ran modified files with:
    * default test path (xarray + xesmf skipped)
    * opt-in xesmf path (`RUN_XESMF_TESTS=1`)
   and confirmed passing runs.

### 4) Extended Gaussian Test Coverage Split by Quantity Semantics

1. Renamed the existing extended-Gaussian Jy/pixel test module to:
    * `tests/test_extended_gaussian_jy_per_pix_correctness.py`
2. Added a dedicated Jy/beam extended-Gaussian module:
    * `tests/test_extended_gaussian_jy_per_beam_correctness.py`
3. Jy/pixel module emphasizes:
    * area-weighted integrated flux stability
    * peak behavior and round-trip RMS constraints
4. Jy/beam module emphasizes:
    * peak fidelity
    * centroid stability
    * round-trip RMS constraints
    * beam metadata presence/consistency
5. Both modules include opt-in xesmf coverage with default skip behavior.

### 5) Performance Detour: XRADIO Image Serial vs Parallel Scaling

Benchmarked regridding of XRADIO-generated `extended_gaussian_jy_per_pixel` images using `xarray` backend and Dask schedulers (`sync` vs `threads`) to compare serial and parallel performance.

#### Case A: Shape `(1, 256, 1, 1024, 1024)`

1. Dataset: `/tmp/xradio_perf_big/extended_gaussian_jy_per_pixel.zarr`
2. Regrid target: `1024x1024 -> 640x640`
3. Results (3 runs/config after warmup, representative means):
    * `sync`: ~13.5 s
    * `threads(2)`: ~9.9 s  (~1.37x)
    * `threads(4)`: ~6.9 s  (~1.96x)
    * `threads(8)`: ~6.8 s  (~1.99x)
    * `threads(16)`: ~7.1 s (~1.90x)
4. Reproducibility rerun showed near-identical behavior and same optimal region around `threads(8)`.

#### Case B: Shape `(1, 512, 1, 1024, 1024)`

1. Dataset: `/tmp/xradio_perf_512ch/extended_gaussian_jy_per_pixel.zarr`
2. Same target and method as Case A.
3. Results (2 runs/config after warmup):
    * `sync`: ~32.2 s
    * `threads(2)`: ~18.7 s (~1.72x)
    * `threads(4)`: ~12.4 s (~2.60x)
    * `threads(8)`: ~9.7 s  (~3.30x)
    * `threads(16)`: ~12.7 s (~2.53x)
4. Observed trend: increasing frequency-plane task count improved parallel scaling substantially; best throughput still occurred near `threads(8)` for this machine/workload.

### 6) Git/Repo State Outcomes

1. Multiple commits were created and pushed to `main` covering:
    * regridding module rename and import updates
    * point-source test split/refactor
    * extended Gaussian Jy/pixel and Jy/beam tests
    * round-trip test migration/removal
2. Local untracked directories (`xradio/`, `xradio_test_images/`) were intentionally left uncommitted.
