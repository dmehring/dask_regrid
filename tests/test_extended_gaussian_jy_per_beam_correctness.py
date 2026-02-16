"""
Extended-Gaussian correctness checks (Jy/beam).

How to run:
  - Default (xarray-only + xesmf skipped):
    python -m pytest -q tests/test_extended_gaussian_jy_per_beam_correctness.py
  - xesmf-only (explicit opt-in):
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_extended_gaussian_jy_per_beam_correctness.py -m xesmf
  - all tests including xesmf:
    RUN_XESMF_TESTS=1 python -m pytest -q tests/test_extended_gaussian_jy_per_beam_correctness.py

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


JYBEAM_FIXTURE = Path("xradio_test_images/extended_gaussian_jy_per_beam.zarr")
JYPIX_FIXTURE = Path("xradio_test_images/extended_gaussian_jy_per_pixel.zarr")


def _can_run_xesmf_tests() -> tuple[bool, str]:
    if os.environ.get("RUN_XESMF_TESTS", "0") != "1":
        return False, "Set RUN_XESMF_TESTS=1 to enable xesmf integration tests"
    try:
        import xesmf  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive import gate
        return False, f"xesmf import failed: {exc}"
    return True, ""


def _source_dataset() -> xr.Dataset:
    if JYBEAM_FIXTURE.exists():
        return xr.open_zarr(JYBEAM_FIXTURE)
    if not JYPIX_FIXTURE.exists():
        pytest.skip(f"Missing fixtures: {JYBEAM_FIXTURE} and {JYPIX_FIXTURE}")

    # Fallback: reuse Gaussian pixels but assign Jy/beam semantics and beam metadata.
    ds = xr.open_zarr(JYPIX_FIXTURE).copy(deep=True)
    ds["SKY"].attrs["units"] = "Jy/beam"
    beam_vals_rad = np.array(
        [np.deg2rad(2.0 / 3600.0), np.deg2rad(1.6 / 3600.0), np.deg2rad(20.0)],
        dtype=np.float64,
    )
    beam = np.broadcast_to(
        beam_vals_rad[None, None, None, :],
        (
            ds.sizes["time"],
            ds.sizes["frequency"],
            ds.sizes["polarization"],
            3,
        ),
    ).copy()
    ds["BEAM_FIT_PARAMS"] = xr.DataArray(
        beam,
        dims=("time", "frequency", "polarization", "beam_params_label"),
        coords={
            "time": ds.coords["time"],
            "frequency": ds.coords["frequency"],
            "polarization": ds.coords["polarization"],
            "beam_params_label": np.array(["major", "minor", "pa"], dtype=object),
        },
    )
    ds["beam_params_label"].attrs["units"] = "rad"
    ds["BEAM_FIT_PARAMS"].attrs["units"] = "rad"
    return ds


def _source() -> xr.DataArray:
    ds = _source_dataset()
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


def _run_identity_checks(backend: str) -> None:
    src = _source()
    out = _regrid(src, backend=backend, new_l=src["l"].values, new_m=src["m"].values)
    a = src.isel(frequency=0).values
    b = out.isel(frequency=0).values
    assert np.nanmax(np.abs(a - b)) <= 1e-12, (
        f"{backend} identity regrid changed Gaussian pixels; max_abs_diff={float(np.nanmax(np.abs(a-b))):.3e}"
    )


def _run_resample_checks(backend: str, n_new: int) -> None:
    src = _source()
    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_new)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_new)
    out = _regrid(src, backend=backend, new_l=new_l, new_m=new_m)
    a = src.isel(frequency=0).values
    b = out.isel(frequency=0).values

    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    cxl, cxm = _centroid_px(b, out["l"].values, out["m"].values)

    # For Jy/beam, emphasize morphology/peak/position over integrated flux.
    assert peak_ratio >= 0.99, (
        f"{backend} Jy/beam resample peak attenuation too large; n_new={n_new}, peak_ratio={peak_ratio:.9f}"
    )
    assert np.nanmin(b) >= 0.0, (
        f"{backend} Jy/beam resample introduced negative values; n_new={n_new}, min={float(np.nanmin(b)):.12g}"
    )
    assert abs(cxl) <= 0.05 and abs(cxm) <= 0.05, (
        f"{backend} Jy/beam resample centroid drift too large; n_new={n_new}, centroid_px=({cxl:.6f}, {cxm:.6f})"
    )


def _run_round_trip_checks(backend: str, n_mid: int) -> None:
    src = _source()
    new_l = np.linspace(float(src["l"].min()), float(src["l"].max()), n_mid)
    new_m = np.linspace(float(src["m"].min()), float(src["m"].max()), n_mid)
    mid = _regrid(src, backend=backend, new_l=new_l, new_m=new_m)
    rt = _regrid(mid, backend=backend, new_l=src["l"].values, new_m=src["m"].values)

    a = src.isel(frequency=0).values
    b = rt.isel(frequency=0).values
    peak_ratio = float(np.nanmax(b) / (np.nanmax(a) + 1e-12))
    rms = float(np.sqrt(np.nanmean((b - a) ** 2)))
    cxl, cxm = _centroid_px(b, rt["l"].values, rt["m"].values)

    assert peak_ratio >= 0.985, (
        f"{backend} Jy/beam round-trip peak attenuation too large; n_mid={n_mid}, peak_ratio={peak_ratio:.9f}"
    )
    assert rms <= 7e-3, (
        f"{backend} Jy/beam round-trip RMS residual too large; n_mid={n_mid}, rms={rms:.9g}"
    )
    assert abs(cxl) <= 0.05 and abs(cxm) <= 0.05, (
        f"{backend} Jy/beam round-trip centroid drift too large; n_mid={n_mid}, centroid_px=({cxl:.6f}, {cxm:.6f})"
    )


def test_extended_gaussian_jy_per_beam_fixture_sanity() -> None:
    ds = _source_dataset()
    src = ds["SKY"].isel(time=0, polarization=0)
    a = src.isel(frequency=0).values

    assert src.attrs.get("units") == "Jy/beam", (
        f"Expected units Jy/beam, got {src.attrs.get('units')!r}"
    )
    # Beam metadata should be carried in BEAM_FIT_PARAMS, not SKY attrs.
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
    assert np.isfinite(a).any(), "Fixture has no finite pixels"
    assert float(np.nanmax(a)) > 0.0, "Fixture peak must be positive"
    assert float(np.nanmin(a)) >= 0.0, "Fixture should be non-negative"


def test_extended_gaussian_jy_per_beam_identity_xarray() -> None:
    _run_identity_checks("xarray")


@pytest.mark.parametrize("n_new", [40, 80])
def test_extended_gaussian_jy_per_beam_resample_xarray(n_new: int) -> None:
    _run_resample_checks("xarray", n_new=n_new)


@pytest.mark.parametrize("n_mid", [40, 80])
def test_extended_gaussian_jy_per_beam_round_trip_xarray(n_mid: int) -> None:
    _run_round_trip_checks("xarray", n_mid=n_mid)


@pytest.mark.xesmf
def test_extended_gaussian_jy_per_beam_identity_xesmf() -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)
    _run_identity_checks("xesmf")


@pytest.mark.xesmf
@pytest.mark.parametrize("n_new", [40, 80])
def test_extended_gaussian_jy_per_beam_resample_xesmf(n_new: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)
    _run_resample_checks("xesmf", n_new=n_new)


@pytest.mark.xesmf
@pytest.mark.parametrize("n_mid", [40, 80])
def test_extended_gaussian_jy_per_beam_round_trip_xesmf(n_mid: int) -> None:
    ok, reason = _can_run_xesmf_tests()
    if not ok:
        pytest.skip(reason)
    _run_round_trip_checks("xesmf", n_mid=n_mid)
