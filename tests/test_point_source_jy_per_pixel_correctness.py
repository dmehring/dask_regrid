"""
Point-source correctness checks for Jy/pixel semantics.

These tests focus on area-weighted integrated flux behavior for a point source.

How to run:
  - Default (xarray-only + xesmf skipped):
    python -m pytest -q tests/test_point_source_jy_per_pixel_correctness.py
  - xesmf-only (explicit opt-in):
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_point_source_jy_per_pixel_correctness.py -m xesmf
  - all tests including xesmf:
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_point_source_jy_per_pixel_correctness.py

Why xesmf tests are skipped by default:
  - xesmf/ESMF can initialize MPI/UCX internals that probe network interfaces/sockets.
  - In restricted environments, this may fail or hard-abort the Python process.
  - To keep default unit-test runs robust, xesmf tests are gated by RUN_XESMF_TESTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from regrid_2d import regrid_2d_planes


JYPIX_FIXTURE_PATH = Path("xradio_test_images/point_source_center_jy_per_pixel.zarr")
FALLBACK_FIXTURE_PATH = Path("xradio_test_images/point_source_center_jy_per_beam.zarr")


def _load_jy_per_pixel_fixture() -> xr.Dataset:
    if JYPIX_FIXTURE_PATH.exists():
        return xr.open_zarr(JYPIX_FIXTURE_PATH)
    if not FALLBACK_FIXTURE_PATH.exists():
        pytest.skip(
            f"Missing fixtures: {JYPIX_FIXTURE_PATH} and fallback {FALLBACK_FIXTURE_PATH}"
        )

    # Fallback for current fixture set: reuse point-source values but override unit semantics.
    ds = xr.open_zarr(FALLBACK_FIXTURE_PATH).copy(deep=True)
    ds["SKY"].attrs["units"] = "Jy/pixel"
    if "BEAM_FIT_PARAMS" in ds:
        ds = ds.drop_vars("BEAM_FIT_PARAMS")
    return ds


def _source_plane(ds: xr.Dataset) -> xr.DataArray:
    # Keep frequency as independent dimension for regrid_2d_planes.
    return ds["SKY"].isel(time=0, polarization=0)


def _can_run_xesmf_tests() -> tuple[bool, str]:
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


def _pixel_area(coord_a: np.ndarray, coord_b: np.ndarray) -> float:
    da = max(float(np.nanmedian(np.abs(np.diff(coord_a)))), 1e-12)
    db = max(float(np.nanmedian(np.abs(np.diff(coord_b)))), 1e-12)
    return da * db


def _integrated_flux(arr2d: np.ndarray, coord_a: np.ndarray, coord_b: np.ndarray) -> float:
    return float(np.nansum(arr2d) * _pixel_area(coord_a, coord_b))


def _centroid_in_pixel_units(
    arr2d: np.ndarray, coord_a: np.ndarray, coord_b: np.ndarray
) -> tuple[float, float]:
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
    da = max(float(np.nanmedian(np.abs(np.diff(coord_a)))), 1e-12)
    db = max(float(np.nanmedian(np.abs(np.diff(coord_b)))), 1e-12)
    return ca / da, cb / db


def test_jy_per_pixel_fixture_sanity() -> None:
    ds = _load_jy_per_pixel_fixture()
    sky = ds["SKY"].isel(time=0, frequency=0, polarization=0).values

    assert ds["SKY"].attrs.get("units") == "Jy/pixel", (
        f"Expected SKY units 'Jy/pixel', got {ds['SKY'].attrs.get('units')!r}"
    )
    assert float(np.nansum(sky)) == pytest.approx(1.0), (
        f"Fixture total pixel-sum should be 1.0, got {float(np.nansum(sky)):.12g}"
    )
    assert float(np.nanmax(sky)) == pytest.approx(1.0), (
        f"Fixture peak should be 1.0, got {float(np.nanmax(sky)):.12g}"
    )


def test_jy_per_pixel_identity_regrid_preserves_integrated_flux() -> None:
    ds = _load_jy_per_pixel_fixture()
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
    src_int = _integrated_flux(src0, src["l"].values, src["m"].values)
    out_int = _integrated_flux(out0, out["l"].values, out["m"].values)

    assert out_int == pytest.approx(src_int, rel=1e-12, abs=1e-12), (
        "Identity regrid should preserve area-weighted integrated flux for Jy/pixel; "
        f"source={src_int:.12g}, regridded={out_int:.12g}"
    )
    assert np.nanmin(out0) >= 0.0, (
        f"Identity regrid introduced negative values; min={float(np.nanmin(out0)):.12g}"
    )


@pytest.mark.parametrize("n_new", [40, 80])
def test_jy_per_pixel_resampled_regrid_flux_and_centroid_stability(n_new: int) -> None:
    ds = _load_jy_per_pixel_fixture()
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

    src0 = src.isel(frequency=0).values
    out0 = out.isel(frequency=0).values
    src_int = _integrated_flux(src0, src["l"].values, src["m"].values)
    out_int = _integrated_flux(out0, out["l"].values, out["m"].values)

    # Linear interpolation is not exactly conservative; use bounded tolerance.
    assert out_int == pytest.approx(src_int, rel=0.30), (
        "Resampled regrid should keep Jy/pixel integrated flux within 30%; "
        f"n_new={n_new}, source={src_int:.12g}, regridded={out_int:.12g}, "
        f"ratio={out_int / src_int:.6f}"
    )
    assert np.nanmin(out0) >= 0.0, (
        f"Resampled regrid (n_new={n_new}) introduced negative values; "
        f"min={float(np.nanmin(out0)):.12g}"
    )

    cx_px, cy_px = _centroid_in_pixel_units(out0, out["l"].values, out["m"].values)
    assert abs(cx_px) <= 0.5, (
        f"Resampled regrid (n_new={n_new}) shifted centroid too far in dim l; "
        f"cx_px={cx_px:.6f}, limit=0.5"
    )
    assert abs(cy_px) <= 0.5, (
        f"Resampled regrid (n_new={n_new}) shifted centroid too far in dim m; "
        f"cy_px={cy_px:.6f}, limit=0.5"
    )


@pytest.mark.xesmf
def test_xesmf_jy_per_pixel_identity_regrid_preserves_integrated_flux() -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    ds = _load_jy_per_pixel_fixture()
    src = _source_plane(ds)
    out = _regrid_backend(
        src,
        backend="xesmf",
        new_a=src["l"].values,
        new_b=src["m"].values,
    )

    src0 = src.isel(frequency=0).values
    out0 = out.isel(frequency=0).values
    src_int = _integrated_flux(src0, src["l"].values, src["m"].values)
    out_int = _integrated_flux(out0, out["l"].values, out["m"].values)

    assert out_int == pytest.approx(src_int, rel=1e-12, abs=1e-12), (
        "xESMF identity regrid should preserve area-weighted integrated flux for Jy/pixel; "
        f"source={src_int:.12g}, regridded={out_int:.12g}"
    )
    assert np.nanmin(out0) >= 0.0, (
        f"xESMF identity regrid introduced negative values; min={float(np.nanmin(out0)):.12g}"
    )


@pytest.mark.xesmf
@pytest.mark.parametrize("n_new", [40, 80])
def test_xesmf_jy_per_pixel_resampled_regrid_flux_and_centroid_stability(
    n_new: int,
) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    ds = _load_jy_per_pixel_fixture()
    src = _source_plane(ds)

    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_new)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_new)
    out = _regrid_backend(src, backend="xesmf", new_a=new_l, new_b=new_m)

    src0 = src.isel(frequency=0).values
    out0 = out.isel(frequency=0).values
    src_int = _integrated_flux(src0, src["l"].values, src["m"].values)
    out_int = _integrated_flux(out0, out["l"].values, out["m"].values)

    # Bilinear interpolation is not strictly conservative; use bounded tolerance.
    assert out_int == pytest.approx(src_int, rel=0.30), (
        "xESMF resampled regrid should keep Jy/pixel integrated flux within 30%; "
        f"n_new={n_new}, source={src_int:.12g}, regridded={out_int:.12g}, "
        f"ratio={out_int / src_int:.6f}"
    )
    assert np.nanmin(out0) >= 0.0, (
        f"xESMF resampled regrid (n_new={n_new}) introduced negative values; "
        f"min={float(np.nanmin(out0)):.12g}"
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
def test_round_trip_jy_per_pixel_xarray_integrated_flux_stability(n_mid: int) -> None:
    ds = _load_jy_per_pixel_fixture()
    src = _source_plane(ds)
    rt = _round_trip(src, backend="xarray", n_mid=n_mid)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values
    flux_ratio = float(np.nansum(b) / (np.nansum(a) + 1e-12))
    cxl, cxm = _centroid_in_pixel_units(b, rt["l"].values, rt["m"].values)

    assert 0.70 <= flux_ratio <= 1.30, (
        f"Round-trip Jy/pixel integrated flux ratio out of bounds; n_mid={n_mid}, flux_ratio={flux_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"Round-trip Jy/pixel introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"Round-trip Jy/pixel centroid drift too large; n_mid={n_mid}, centroid_px=({cxl:.6f}, {cxm:.6f})"
    )


@pytest.mark.xesmf
@pytest.mark.parametrize("n_mid", [40, 80])
def test_round_trip_jy_per_pixel_xesmf_integrated_flux_stability(n_mid: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)

    ds = _load_jy_per_pixel_fixture()
    src = _source_plane(ds)
    rt = _round_trip(src, backend="xesmf", n_mid=n_mid)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values
    flux_ratio = float(np.nansum(b) / (np.nansum(a) + 1e-12))
    cxl, cxm = _centroid_in_pixel_units(b, rt["l"].values, rt["m"].values)

    assert 0.70 <= flux_ratio <= 1.30, (
        f"xESMF round-trip Jy/pixel integrated flux ratio out of bounds; n_mid={n_mid}, flux_ratio={flux_ratio:.6f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"xESMF round-trip Jy/pixel introduced negative values; n_mid={n_mid}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.30 and abs(cxm) <= 0.30, (
        f"xESMF round-trip Jy/pixel centroid drift too large; n_mid={n_mid}, centroid_px=({cxl:.6f}, {cxm:.6f})"
    )
