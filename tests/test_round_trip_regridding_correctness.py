"""
Round-trip correctness tests:
  original grid -> intermediate grid -> original grid.

How to run:
  - Default (xarray-only + xesmf skipped):
    python -m pytest -q tests/test_round_trip_regridding_correctness.py
  - xesmf-only (explicit opt-in):
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_round_trip_regridding_correctness.py -m xesmf
  - all tests including xesmf:
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_round_trip_regridding_correctness.py

Why xesmf tests are skipped by default:
  - xesmf/ESMF may initialize MPI/UCX internals that probe sockets/interfaces.
  - In restricted environments this can fail or hard-abort Python.
  - We gate xesmf tests behind RUN_XESMF_TESTS=1 for robust default runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from regrid_3d import regrid_2d_planes


JYBEAM_FIXTURE = Path("xradio_test_images/point_source_center_jy_per_beam.zarr")
JYPIX_FIXTURE = Path("xradio_test_images/point_source_center_jy_per_pixel.zarr")


def _can_run_xesmf_tests() -> tuple[bool, str]:
    if os.environ.get("RUN_XESMF_TESTS", "0") != "1":
        return False, "Set RUN_XESMF_TESTS=1 to enable xesmf integration tests"
    try:
        import xesmf  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return False, f"xesmf import failed: {exc}"
    return True, ""


def _load_jybeam_source() -> xr.DataArray:
    if not JYBEAM_FIXTURE.exists():
        pytest.skip(f"Missing fixture: {JYBEAM_FIXTURE}")
    ds = xr.open_zarr(JYBEAM_FIXTURE)
    return ds["SKY"].isel(time=0, polarization=0)


def _load_jypix_source() -> xr.DataArray:
    if JYPIX_FIXTURE.exists():
        ds = xr.open_zarr(JYPIX_FIXTURE)
        return ds["SKY"].isel(time=0, polarization=0)
    if not JYBEAM_FIXTURE.exists():
        pytest.skip(f"Missing fixtures: {JYPIX_FIXTURE} and {JYBEAM_FIXTURE}")
    ds = xr.open_zarr(JYBEAM_FIXTURE).copy(deep=True)
    ds["SKY"].attrs["units"] = "Jy/pixel"
    for k in ("beam_major_arcsec", "beam_minor_arcsec", "beam_pa_deg"):
        ds["SKY"].attrs.pop(k, None)
    return ds["SKY"].isel(time=0, polarization=0)


def _regrid(
    src: xr.DataArray,
    *,
    backend: str,
    new_l: np.ndarray,
    new_m: np.ndarray,
) -> xr.DataArray:
    return regrid_2d_planes(
        src,
        dim_a="l",
        dim_b="m",
        new_coord_a=new_l,
        new_coord_b=new_m,
        regridder_name=backend,
        method="linear",
    ).compute()


def _round_trip(src: xr.DataArray, *, backend: str, n_mid: int) -> xr.DataArray:
    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_mid)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_mid)
    mid = _regrid(src, backend=backend, new_l=new_l, new_m=new_m)
    return _regrid(
        mid,
        backend=backend,
        new_l=src["l"].values,
        new_m=src["m"].values,
    )


def _centroid_px(arr2d: np.ndarray, coord_l: np.ndarray, coord_m: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(arr2d)
    if not valid.any():
        return float("nan"), float("nan")
    w = np.abs(np.where(valid, arr2d, 0.0))
    total = float(w.sum())
    if total == 0.0:
        return float("nan"), float("nan")
    wl = w.sum(axis=1)
    wm = w.sum(axis=0)
    l0 = float(np.sum(wl * coord_l) / total)
    m0 = float(np.sum(wm * coord_m) / total)
    dl = max(float(np.nanmedian(np.abs(np.diff(coord_l)))), 1e-12)
    dm = max(float(np.nanmedian(np.abs(np.diff(coord_m)))), 1e-12)
    return l0 / dl, m0 / dm


@pytest.mark.parametrize("n_mid", [40, 80])
def test_round_trip_jy_per_beam_xarray_stability(n_mid: int) -> None:
    src = _load_jybeam_source()
    rt = _round_trip(src, backend="xarray", n_mid=n_mid)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values

    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    cxl, cxm = _centroid_px(b, rt["l"].values, rt["m"].values)

    # Round trip with linear interpolation is lossy for point sources.
    # We enforce bounded degradation and bounded centroid drift.
    assert peak_ratio >= 0.25, (
        f"Round-trip Jy/beam peak dropped too much; n_mid={n_mid}, peak_ratio={peak_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"Round-trip Jy/beam introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"Round-trip Jy/beam centroid drift too large; n_mid={n_mid}, "
        f"centroid_px=({cxl:.6f}, {cxm:.6f})"
    )


@pytest.mark.parametrize("n_mid", [40, 80])
def test_round_trip_jy_per_pixel_xarray_integrated_flux_stability(n_mid: int) -> None:
    src = _load_jypix_source()
    rt = _round_trip(src, backend="xarray", n_mid=n_mid)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values

    # Same original/final grid => pixel-area factors cancel in ratio.
    flux_ratio = float(np.nansum(b) / (np.nansum(a) + 1e-12))
    cxl, cxm = _centroid_px(b, rt["l"].values, rt["m"].values)

    assert 0.70 <= flux_ratio <= 1.30, (
        f"Round-trip Jy/pixel integrated flux ratio out of bounds; "
        f"n_mid={n_mid}, flux_ratio={flux_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"Round-trip Jy/pixel introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"Round-trip Jy/pixel centroid drift too large; n_mid={n_mid}, "
        f"centroid_px=({cxl:.6f}, {cxm:.6f})"
    )


@pytest.mark.xesmf
@pytest.mark.parametrize("n_mid", [40, 80])
def test_round_trip_jy_per_beam_xesmf_stability(n_mid: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    src = _load_jybeam_source()
    rt = _round_trip(src, backend="xesmf", n_mid=n_mid)
    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values

    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    cxl, cxm = _centroid_px(b, rt["l"].values, rt["m"].values)

    assert peak_ratio >= 0.25, (
        f"xESMF round-trip Jy/beam peak dropped too much; n_mid={n_mid}, peak_ratio={peak_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"xESMF round-trip Jy/beam introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"xESMF round-trip Jy/beam centroid drift too large; n_mid={n_mid}, "
        f"centroid_px=({cxl:.6f}, {cxm:.6f})"
    )


@pytest.mark.xesmf
@pytest.mark.parametrize("n_mid", [40, 80])
def test_round_trip_jy_per_pixel_xesmf_integrated_flux_stability(n_mid: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    src = _load_jypix_source()
    rt = _round_trip(src, backend="xesmf", n_mid=n_mid)
    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values

    flux_ratio = float(np.nansum(b) / (np.nansum(a) + 1e-12))
    cxl, cxm = _centroid_px(b, rt["l"].values, rt["m"].values)

    assert 0.70 <= flux_ratio <= 1.30, (
        f"xESMF round-trip Jy/pixel integrated flux ratio out of bounds; "
        f"n_mid={n_mid}, flux_ratio={flux_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"xESMF round-trip Jy/pixel introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"xESMF round-trip Jy/pixel centroid drift too large; n_mid={n_mid}, "
        f"centroid_px=({cxl:.6f}, {cxm:.6f})"
    )

