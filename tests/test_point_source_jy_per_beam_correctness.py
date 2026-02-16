"""
Point-source correctness checks for regridding backends.

How to run:
  - Default (xarray-only + xesmf skipped):
    python -m pytest -q tests/test_point_source_jy_per_beam_correctness.py
  - xesmf-only (explicit opt-in):
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_point_source_jy_per_beam_correctness.py -m xesmf
  - all tests including xesmf:
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_point_source_jy_per_beam_correctness.py

Why xesmf tests may not run:
  - xesmf/ESMF can initialize MPI/UCX internals that open/probe sockets and interfaces.
  - In restricted environments (CI sandboxes, locked-down containers), that can fail
    or hard-abort the Python process.
  - For safety, xesmf tests require RUN_XESMF_TESTS=1 and are skipped otherwise.

Shell commands used during development in this repo:
```bash
# xarray-only suite
python -m pytest -q tests/test_point_source_jy_per_beam_correctness.py

# opt-in xesmf suite
RUN_XESMF_TESTS=1 python -m pytest -q tests/test_point_source_jy_per_beam_correctness.py -m xesmf

# direct backend probe used to inspect xesmf behavior
python - <<'PY'
import numpy as np, xarray as xr
from regrid_2d import regrid_2d_planes
src = xr.open_zarr('xradio_test_images/point_source_center_jy_per_beam.zarr')['SKY'].isel(time=0, polarization=0)
src2 = src.rename({'l': 'lat', 'm': 'lon'})
for n in [64, 40, 80]:
    new = np.linspace(float(src2.lat.min()), float(src2.lat.max()), n)
    out = regrid_2d_planes(src2, 'lat', 'lon', new, new, regridder_name='xesmf', method='linear').compute()
    b = out.isel(frequency=0).values
    print(n, float(np.nansum(b)), float(np.nanmax(b)))
PY
```
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from regrid_2d import regrid_2d_planes


FIXTURE_PATH = Path("xradio_test_images/point_source_center_jy_per_beam.zarr")


def _require_fixture() -> xr.Dataset:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Missing fixture: {FIXTURE_PATH}")
    return xr.open_zarr(FIXTURE_PATH)


def _source_plane(ds: xr.Dataset) -> xr.DataArray:
    # Keep frequency as independent dimension for regrid_2d_planes.
    return ds["SKY"].isel(time=0, polarization=0)


def _can_run_xesmf_tests() -> tuple[bool, str]:
    # xESMF can hard-abort in restricted runtimes (MPI/UCX socket setup).
    # Require explicit opt-in so default unit-test runs remain robust.
    if os.environ.get("RUN_XESMF_TESTS", "0") != "1":
        return False, "Set RUN_XESMF_TESTS=1 to enable xesmf integration tests"
    try:
        import xesmf  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive import gate
        return False, f"xesmf import failed: {exc}"
    return True, ""


def _regrid_backend(
    src: xr.DataArray,
    *,
    backend: str,
    new_a: np.ndarray,
    new_b: np.ndarray,
) -> xr.DataArray:
    return regrid_2d_planes(
        src,
        dim_a="l",
        dim_b="m",
        new_coord_a=new_a,
        new_coord_b=new_b,
        regridder_name=backend,
        method="linear",
    ).compute()


def _round_trip(src: xr.DataArray, *, backend: str, n_mid: int) -> xr.DataArray:
    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_mid)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_mid)
    mid = _regrid_backend(src, backend=backend, new_a=new_l, new_b=new_m)
    return _regrid_backend(
        mid,
        backend=backend,
        new_a=src["l"].values,
        new_b=src["m"].values,
    )


def _centroid_in_pixel_units(arr2d: np.ndarray, coord_a: np.ndarray, coord_b: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(arr2d)
    if not valid.any():
        return float("nan"), float("nan")
    w = np.abs(np.where(valid, arr2d, 0.0))
    total = float(w.sum())
    if total == 0.0:
        return float("nan"), float("nan")
    wa = w.sum(axis=1)
    wb = w.sum(axis=0)
    ca = float(np.sum(wa * coord_a) / total)
    cb = float(np.sum(wb * coord_b) / total)
    # Convert to pixel units using median spacing.
    da = max(float(np.nanmedian(np.abs(np.diff(coord_a)))), 1e-12)
    db = max(float(np.nanmedian(np.abs(np.diff(coord_b)))), 1e-12)
    return ca / da, cb / db


def test_point_source_fixture_sanity() -> None:
    ds = _require_fixture()
    sky = ds["SKY"].isel(time=0, frequency=0, polarization=0).values

    # Baseline fixture assumptions: point source image in Jy/beam.
    assert ds["SKY"].attrs.get("units") == "Jy/beam", (
        f"Expected SKY units 'Jy/beam' for point-source fixture, "
        f"got {ds['SKY'].attrs.get('units')!r}"
    )
    assert "BEAM_FIT_PARAMS" in ds.data_vars, (
        "Expected BEAM_FIT_PARAMS variable for Jy/beam fixture, but it is missing"
    )
    beam = ds["BEAM_FIT_PARAMS"]
    assert tuple(beam.dims) == ("time", "frequency", "polarization", "beam_params_label"), (
        f"Unexpected BEAM_FIT_PARAMS dims: {beam.dims}"
    )
    labels = [str(x) for x in beam["beam_params_label"].values.tolist()]
    assert labels == ["major", "minor", "pa"], (
        f"Unexpected beam_params_label values: {labels}"
    )
    assert "units" in beam["beam_params_label"].attrs, (
        "Expected beam_params_label coordinate to define angular units"
    )
    # Source construction sets one unit-flux pixel at center.
    assert float(np.nansum(sky)) == pytest.approx(1.0), (
        f"Fixture total flux should be 1.0, got {float(np.nansum(sky)):.12g}"
    )
    assert float(np.nanmax(sky)) == pytest.approx(1.0), (
        f"Fixture peak should be 1.0, got {float(np.nanmax(sky)):.12g}"
    )
    max_loc = np.unravel_index(np.nanargmax(sky), sky.shape)
    # Peak should be exactly centered in the synthetic fixture.
    assert max_loc == (sky.shape[0] // 2, sky.shape[1] // 2), (
        f"Point-source peak expected at image center, got index {max_loc}"
    )


def test_identity_regrid_preserves_peak_sum_and_center() -> None:
    ds = _require_fixture()
    src = _source_plane(ds)

    out = regrid_2d_planes(
        src,
        dim_a="l",
        dim_b="m",
        new_coord_a=src["l"].values,
        new_coord_b=src["m"].values,
        regridder_name="xarray",
        method="linear",
    ).compute()

    src0 = src.isel(frequency=0).values
    out0 = out.isel(frequency=0).values
    src_max = np.unravel_index(np.nanargmax(src0), src0.shape)
    out_max = np.unravel_index(np.nanargmax(out0), out0.shape)

    # Identity regrid (same coordinates) should preserve total flux and peak exactly.
    assert float(np.nansum(out0)) == pytest.approx(
        float(np.nansum(src0)), rel=1e-12, abs=1e-12
    ), (
        "Identity regrid should preserve total flux; "
        f"source={float(np.nansum(src0)):.12g}, regridded={float(np.nansum(out0)):.12g}"
    )
    assert float(np.nanmax(out0)) == pytest.approx(
        float(np.nanmax(src0)), rel=1e-12, abs=1e-12
    ), (
        "Identity regrid should preserve peak amplitude; "
        f"source={float(np.nanmax(src0)):.12g}, regridded={float(np.nanmax(out0)):.12g}"
    )
    # Peak location should not drift on identity transform.
    assert out_max == src_max, (
        f"Identity regrid shifted peak location from {src_max} to {out_max}"
    )
    # Linear interpolation should not create negative values from non-negative input.
    assert np.nanmin(out0) >= 0.0, (
        f"Identity regrid introduced negative values; min={float(np.nanmin(out0)):.12g}"
    )
    # Mask can expand under interpolation; it should not shrink.
    assert int(np.isnan(out0).sum()) >= int(np.isnan(src0).sum()), (
        "Identity regrid unexpectedly reduced NaN-mask footprint; "
        f"source_nans={int(np.isnan(src0).sum())}, out_nans={int(np.isnan(out0).sum())}"
    )


@pytest.mark.parametrize("n_new", [40, 80])
def test_resampled_regrid_keeps_source_centered_and_nonnegative(n_new: int) -> None:
    ds = _require_fixture()
    src = _source_plane(ds)

    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_new)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_new)
    out = regrid_2d_planes(
        src,
        dim_a="l",
        dim_b="m",
        new_coord_a=new_l,
        new_coord_b=new_m,
        regridder_name="xarray",
        method="linear",
    ).compute()

    out0 = out.isel(frequency=0).values
    # Regridding a non-negative point source should stay non-negative.
    assert np.nanmin(out0) >= 0.0, (
        f"Resampled regrid (n_new={n_new}) introduced negative values; "
        f"min={float(np.nanmin(out0)):.12g}"
    )
    assert np.isfinite(np.nanmax(out0)), (
        f"Resampled regrid (n_new={n_new}) produced non-finite peak value"
    )

    cx_px, cy_px = _centroid_in_pixel_units(out0, out["l"].values, out["m"].values)
    # For both downsample and upsample factors, centroid should remain near phase center.
    # 0.5 pixel bound allows interpolation spread without allowing meaningful drift.
    assert abs(cx_px) <= 0.5, (
        f"Resampled regrid (n_new={n_new}) shifted centroid too far in dim l; "
        f"cx_px={cx_px:.6f}, limit=0.5"
    )
    assert abs(cy_px) <= 0.5, (
        f"Resampled regrid (n_new={n_new}) shifted centroid too far in dim m; "
        f"cy_px={cy_px:.6f}, limit=0.5"
    )


@pytest.mark.xesmf
def test_xesmf_identity_regrid_preserves_peak_sum_and_center() -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    ds = _require_fixture()
    src = _source_plane(ds)
    out = _regrid_backend(src, backend="xesmf", new_a=src["l"].values, new_b=src["m"].values)

    src0 = src.isel(frequency=0).values
    out0 = out.isel(frequency=0).values

    assert float(np.nansum(out0)) == pytest.approx(float(np.nansum(src0)), rel=1e-12, abs=1e-12), (
        "xESMF identity regrid should preserve total flux; "
        f"source={float(np.nansum(src0)):.12g}, regridded={float(np.nansum(out0)):.12g}"
    )
    assert float(np.nanmax(out0)) == pytest.approx(float(np.nanmax(src0)), rel=1e-12, abs=1e-12), (
        "xESMF identity regrid should preserve peak amplitude; "
        f"source={float(np.nanmax(src0)):.12g}, regridded={float(np.nanmax(out0)):.12g}"
    )
    assert np.nanmin(out0) >= 0.0, (
        f"xESMF identity regrid introduced negative values; min={float(np.nanmin(out0)):.12g}"
    )


@pytest.mark.xesmf
@pytest.mark.parametrize("n_new", [40, 80])
def test_xesmf_resampled_regrid_keeps_source_centered_and_nonnegative(n_new: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    ds = _require_fixture()
    src = _source_plane(ds)

    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_new)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_new)
    out = _regrid_backend(src, backend="xesmf", new_a=new_l, new_b=new_m)
    out0 = out.isel(frequency=0).values

    assert np.nanmin(out0) >= 0.0, (
        f"xESMF resampled regrid (n_new={n_new}) introduced negative values; "
        f"min={float(np.nanmin(out0)):.12g}"
    )
    assert np.isfinite(np.nanmax(out0)), (
        f"xESMF resampled regrid (n_new={n_new}) produced non-finite peak value"
    )

    cx_px, cy_px = _centroid_in_pixel_units(out0, out["l"].values, out["m"].values)
    assert abs(cx_px) <= 0.5, (
        f"xESMF resampled regrid (n_new={n_new}) shifted centroid too far in dim l; "
        f"cx_px={cx_px:.6f}, limit=0.5"
    )
    assert abs(cy_px) <= 0.5, (
        f"xESMF resampled regrid (n_new={n_new}) shifted centroid too far in dim m; "
        f"cy_px={cy_px:.6f}, limit=0.5"
    )


@pytest.mark.parametrize("n_mid", [40, 80])
def test_round_trip_jy_per_beam_xarray_stability(n_mid: int) -> None:
    ds = _require_fixture()
    src = _source_plane(ds)
    rt = _round_trip(src, backend="xarray", n_mid=n_mid)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values
    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    cxl, cxm = _centroid_in_pixel_units(b, rt["l"].values, rt["m"].values)

    assert peak_ratio >= 0.25, (
        f"Round-trip Jy/beam peak dropped too much; n_mid={n_mid}, peak_ratio={peak_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"Round-trip Jy/beam introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"Round-trip Jy/beam centroid drift too large; n_mid={n_mid}, centroid_px=({cxl:.6f}, {cxm:.6f})"
    )


@pytest.mark.xesmf
@pytest.mark.parametrize("n_mid", [40, 80])
def test_round_trip_jy_per_beam_xesmf_stability(n_mid: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    ds = _require_fixture()
    src = _source_plane(ds)
    rt = _round_trip(src, backend="xesmf", n_mid=n_mid)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values
    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    cxl, cxm = _centroid_in_pixel_units(b, rt["l"].values, rt["m"].values)

    assert peak_ratio >= 0.25, (
        f"xESMF round-trip Jy/beam peak dropped too much; n_mid={n_mid}, peak_ratio={peak_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"xESMF round-trip Jy/beam introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"xESMF round-trip Jy/beam centroid drift too large; n_mid={n_mid}, centroid_px=({cxl:.6f}, {cxm:.6f})"
    )
