"""
Extended-Gaussian correctness checks (Jy/pixel).

How to run:
  - Default (xarray-only + xesmf skipped):
    python -m pytest -q tests/test_extended_gaussian_jy_per_pix_correctness.py
  - xesmf-only (explicit opt-in):
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_extended_gaussian_jy_per_pix_correctness.py -m xesmf
  - all tests including xesmf:
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_extended_gaussian_jy_per_pix_correctness.py

Why xesmf tests are skipped by default:
  - xesmf/ESMF may initialize MPI/UCX internals that probe sockets/interfaces.
  - In restricted environments this can fail or hard-abort the Python process.
  - xesmf tests are gated behind RUN_XESMF_TESTS=1 for robust default runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from regrid_2d import regrid_2d_planes


FIXTURE = Path("xradio_test_images/extended_gaussian_jy_per_pixel.zarr")


def _can_run_xesmf_tests() -> tuple[bool, str]:
    if os.environ.get("RUN_XESMF_TESTS", "0") != "1":
        return False, "Set RUN_XESMF_TESTS=1 to enable xesmf integration tests"
    try:
        import xesmf  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive import gate
        return False, f"xesmf import failed: {exc}"
    return True, ""


def _source() -> xr.DataArray:
    if not FIXTURE.exists():
        pytest.skip(f"Missing fixture: {FIXTURE}")
    ds = xr.open_zarr(FIXTURE)
    return ds["SKY"].isel(time=0, polarization=0)


def _regrid(src: xr.DataArray, *, backend: str, new_l: np.ndarray, new_m: np.ndarray) -> xr.DataArray:
    return regrid_2d_planes(
        src,
        dim_a="l",
        dim_b="m",
        new_coord_a=new_l,
        new_coord_b=new_m,
        regridder_name=backend,
        method="linear",
    ).compute()


def _pixel_area(l: np.ndarray, m: np.ndarray) -> float:
    dl = max(float(np.nanmedian(np.abs(np.diff(l)))), 1e-12)
    dm = max(float(np.nanmedian(np.abs(np.diff(m)))), 1e-12)
    return dl * dm


def _integrated_flux(arr2d: np.ndarray, l: np.ndarray, m: np.ndarray) -> float:
    return float(np.nansum(arr2d) * _pixel_area(l, m))


def _run_identity_checks(backend: str) -> None:
    src = _source()
    out = _regrid(src, backend=backend, new_l=src["l"].values, new_m=src["m"].values)

    a = src.isel(frequency=0).values
    b = out.isel(frequency=0).values
    int_a = _integrated_flux(a, src["l"].values, src["m"].values)
    int_b = _integrated_flux(b, out["l"].values, out["m"].values)

    assert np.nanmax(np.abs(a - b)) <= 1e-12, (
        f"{backend} identity regrid changed pixels; max_abs_diff={float(np.nanmax(np.abs(a-b))):.3e}"
    )
    assert int_b == pytest.approx(int_a, rel=1e-12, abs=1e-12), (
        f"{backend} identity regrid changed integrated flux; src={int_a:.12g}, out={int_b:.12g}"
    )


def _run_resample_checks(backend: str, n_new: int) -> None:
    src = _source()
    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_new)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_new)
    out = _regrid(src, backend=backend, new_l=new_l, new_m=new_m)

    a = src.isel(frequency=0).values
    b = out.isel(frequency=0).values
    int_a = _integrated_flux(a, src["l"].values, src["m"].values)
    int_b = _integrated_flux(b, out["l"].values, out["m"].values)

    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    int_ratio = float(int_b / (int_a + 1e-24))

    # Smooth field should preserve area-weighted integrated flux very well.
    assert int_ratio == pytest.approx(1.0, rel=2e-3), (
        f"{backend} resample integrated flux drift too large; n_new={n_new}, ratio={int_ratio:.9f}"
    )
    # Peak can soften slightly when downsampling, but should remain close.
    assert peak_ratio >= 0.99, (
        f"{backend} resample peak attenuation too large; n_new={n_new}, peak_ratio={peak_ratio:.9f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"{backend} resample introduced negative values; n_new={n_new}, min={float(np.nanmin(b)):.12g}"
    )


def _run_round_trip_checks(backend: str, n_mid: int) -> None:
    src = _source()
    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_mid)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_mid)
    mid = _regrid(src, backend=backend, new_l=new_l, new_m=new_m)
    rt = _regrid(mid, backend=backend, new_l=src["l"].values, new_m=src["m"].values)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values
    sum_ratio = float(np.nansum(b) / (np.nansum(a) + 1e-12))
    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    rms = float(np.sqrt(np.nanmean((b - a) ** 2)))

    assert sum_ratio == pytest.approx(1.0, rel=2e-3), (
        f"{backend} round-trip sum drift too large; n_mid={n_mid}, sum_ratio={sum_ratio:.9f}"
    )
    assert peak_ratio >= 0.985, (
        f"{backend} round-trip peak attenuation too large; n_mid={n_mid}, peak_ratio={peak_ratio:.9f}"
    )
    assert rms <= 7e-3, (
        f"{backend} round-trip RMS residual too large; n_mid={n_mid}, rms={rms:.9g}"
    )


def test_extended_gaussian_fixture_sanity() -> None:
    src = _source()
    a = src.isel(frequency=0).values

    assert src.attrs.get("units") == "Jy/pixel", (
        f"Expected units Jy/pixel, got {src.attrs.get('units')!r}"
    )
    assert np.isfinite(a).any(), "Fixture has no finite pixels"
    assert float(np.nanmax(a)) > 0.0, "Fixture peak must be positive"
    assert float(np.nanmin(a)) >= 0.0, "Fixture should be non-negative"


def test_extended_gaussian_identity_xarray() -> None:
    _run_identity_checks("xarray")


@pytest.mark.parametrize("n_new", [40, 80])
def test_extended_gaussian_resample_xarray(n_new: int) -> None:
    _run_resample_checks("xarray", n_new=n_new)


@pytest.mark.parametrize("n_mid", [40, 80])
def test_extended_gaussian_round_trip_xarray(n_mid: int) -> None:
    _run_round_trip_checks("xarray", n_mid=n_mid)


@pytest.mark.xesmf
def test_extended_gaussian_identity_xesmf() -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)
    _run_identity_checks("xesmf")


@pytest.mark.xesmf
@pytest.mark.parametrize("n_new", [40, 80])
def test_extended_gaussian_resample_xesmf(n_new: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)
    _run_resample_checks("xesmf", n_new=n_new)


@pytest.mark.xesmf
@pytest.mark.parametrize("n_mid", [40, 80])
def test_extended_gaussian_round_trip_xesmf(n_mid: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)
    _run_round_trip_checks("xesmf", n_mid=n_mid)
