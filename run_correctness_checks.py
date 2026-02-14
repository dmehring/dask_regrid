"""
Run a correctness-focused regridding comparison and emit a JSON report.

This is a practical scaffold for radio-astro validation workflows where
scientific correctness matters more than runtime speed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from regrid_2d import regrid_2d_planes


def _as_text(x: Any) -> str:
    return str(x).strip()


def _select_variable(ds: xr.Dataset, requested: str | None) -> str:
    if requested:
        if requested not in ds.data_vars:
            raise ValueError(
                f"Variable {requested!r} not found in dataset data vars: {list(ds.data_vars)}"
            )
        return requested
    if "SKY" in ds.data_vars:
        return "SKY"
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(
        "Multiple data variables found. Pass --variable explicitly. "
        f"Available: {list(ds.data_vars)}"
    )


def _infer_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    lower_dims = {d.lower(): d for d in da.dims}
    for a, b in (("l", "m"), ("ra", "dec"), ("lat", "lon")):
        if a in lower_dims and b in lower_dims:
            return lower_dims[a], lower_dims[b]
    raise ValueError(
        "Could not auto-detect spatial dims. Pass --dim-a and --dim-b explicitly. "
        f"Data dims: {da.dims}"
    )


def _detect_quantity_mode(
    ds: xr.Dataset,
    da: xr.DataArray,
    requested_mode: str,
) -> str:
    if requested_mode != "auto":
        return requested_mode

    unit_candidates = []
    for key in ("units", "unit", "bunit"):
        if key in da.attrs:
            unit_candidates.append(_as_text(da.attrs[key]).lower())
    for key in ("units", "unit", "bunit"):
        if key in ds.attrs:
            unit_candidates.append(_as_text(ds.attrs[key]).lower())

    joined = " ".join(unit_candidates)
    if "jy/beam" in joined or "jy beam-1" in joined:
        return "jy_per_beam"
    if "jy/pixel" in joined or "jy pix-1" in joined:
        return "jy_per_pixel"
    return "generic"


def _collect_schema_hints(ds: xr.Dataset, da: xr.DataArray) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset_attrs": {}, "data_var_attrs": {}}
    for k, v in ds.attrs.items():
        kl = str(k).lower()
        if "schema" in kl or "xradio" in kl:
            out["dataset_attrs"][str(k)] = _as_text(v)
    for k, v in da.attrs.items():
        kl = str(k).lower()
        if "schema" in kl or "xradio" in kl or k in ("units", "unit", "bunit"):
            out["data_var_attrs"][str(k)] = _as_text(v)
    out["coord_attrs"] = {}
    for cname, c in da.coords.items():
        keyvals: dict[str, str] = {}
        for k, v in c.attrs.items():
            kl = str(k).lower()
            if (
                "schema" in kl
                or "xradio" in kl
                or k in ("units", "unit", "ctype", "axis", "frame", "name")
            ):
                keyvals[str(k)] = _as_text(v)
        if keyvals:
            out["coord_attrs"][str(cname)] = keyvals
    return out


def _resolve_beam_metadata(
    ds: xr.Dataset,
    da: xr.DataArray,
    quantity_mode: str,
    beam_major_arcsec: float | None,
    beam_minor_arcsec: float | None,
    beam_pa_deg: float | None,
) -> dict[str, float | None]:
    major = beam_major_arcsec
    minor = beam_minor_arcsec
    pa = beam_pa_deg

    # Best-effort extraction from attrs if CLI did not provide values.
    if major is None:
        for key in ("beam_major_arcsec", "bmaj_arcsec", "beam_major"):
            if key in da.attrs:
                major = float(da.attrs[key])
                break
            if key in ds.attrs:
                major = float(ds.attrs[key])
                break
    if minor is None:
        for key in ("beam_minor_arcsec", "bmin_arcsec", "beam_minor"):
            if key in da.attrs:
                minor = float(da.attrs[key])
                break
            if key in ds.attrs:
                minor = float(ds.attrs[key])
                break
    if pa is None:
        for key in ("beam_pa_deg", "bpa_deg", "beam_pa"):
            if key in da.attrs:
                pa = float(da.attrs[key])
                break
            if key in ds.attrs:
                pa = float(ds.attrs[key])
                break

    if quantity_mode == "jy_per_beam":
        if major is None or minor is None:
            raise ValueError(
                "Beam metadata required for Jy/beam workflows. "
                "Pass --beam-major-arcsec/--beam-minor-arcsec or provide matching attrs."
            )
        if major <= 0 or minor <= 0:
            raise ValueError("Beam major/minor must be positive.")

    return {"major_arcsec": major, "minor_arcsec": minor, "pa_deg": pa}


def _infer_slice_dim(da: xr.DataArray, dim_a: str, dim_b: str) -> str | None:
    for dim in da.dims:
        if dim not in (dim_a, dim_b):
            return dim
    return None


def _subset_data(da: xr.DataArray, slice_dim: str | None, max_slices: int) -> xr.DataArray:
    if slice_dim is None or slice_dim not in da.dims:
        return da
    n = da.sizes[slice_dim]
    if n <= max_slices:
        return da
    return da.isel({slice_dim: slice(0, max_slices)})


def _target_size(n_src: int, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return max(8, int(round(n_src * 0.6)))


def _build_target_coords(
    da: xr.DataArray, dim_a: str, dim_b: str, n_a_new: int | None, n_b_new: int | None
) -> tuple[np.ndarray, np.ndarray]:
    n_a = _target_size(da.sizes[dim_a], n_a_new)
    n_b = _target_size(da.sizes[dim_b], n_b_new)
    a0 = float(da.coords[dim_a].min().item())
    a1 = float(da.coords[dim_a].max().item())
    b0 = float(da.coords[dim_b].min().item())
    b1 = float(da.coords[dim_b].max().item())
    return np.linspace(a0, a1, n_a), np.linspace(b0, b1, n_b)


def _run_backend(
    da: xr.DataArray,
    dim_a: str,
    dim_b: str,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    backend: str,
    method: str,
    fill_value: float | None,
) -> xr.DataArray:
    return regrid_2d_planes(
        da,
        dim_a=dim_a,
        dim_b=dim_b,
        new_coord_a=new_coord_a,
        new_coord_b=new_coord_b,
        regridder_name=backend,
        method=method,
        fill_value=fill_value,
    )


def _weighted_centroid_2d(
    arr2d: np.ndarray, coord_a: np.ndarray, coord_b: np.ndarray, eps: float
) -> tuple[float, float]:
    valid = np.isfinite(arr2d)
    if not valid.any():
        return float("nan"), float("nan")
    weights = np.abs(np.where(valid, arr2d, 0.0))
    total = float(weights.sum())
    if total <= eps:
        return float("nan"), float("nan")

    # weights_a shape: (n_a,), weights_b shape: (n_b,)
    weights_a = weights.sum(axis=1)
    weights_b = weights.sum(axis=0)
    ca = float(np.sum(weights_a * coord_a) / total)
    cb = float(np.sum(weights_b * coord_b) / total)
    return ca, cb


def _pixel_scales(coord_a: np.ndarray, coord_b: np.ndarray) -> tuple[float, float]:
    da = np.diff(coord_a)
    db = np.diff(coord_b)
    scale_a = float(np.nanmedian(np.abs(da))) if da.size else 1.0
    scale_b = float(np.nanmedian(np.abs(db))) if db.size else 1.0
    return max(scale_a, 1e-12), max(scale_b, 1e-12)


def _as_list(x: Any) -> list[float]:
    arr = np.asarray(x).astype(float)
    return arr.reshape(-1).tolist()


def _compute_metrics(
    ref: xr.DataArray,
    cand: xr.DataArray,
    dim_a: str,
    dim_b: str,
    eps: float,
) -> dict[str, Any]:
    ref, cand = xr.align(ref, cand, join="exact")

    spatial_dims = (dim_a, dim_b)
    other_dims = [d for d in ref.dims if d not in spatial_dims]

    ref_vals = ref.values
    cand_vals = cand.values

    # Flatten non-spatial dimensions into batch axis for per-slice metrics.
    if other_dims:
        ref_t = ref.transpose(*other_dims, *spatial_dims).values
        cand_t = cand.transpose(*other_dims, *spatial_dims).values
        batch = int(np.prod(ref_t.shape[:-2]))
        ref_slices = ref_t.reshape(batch, ref_t.shape[-2], ref_t.shape[-1])
        cand_slices = cand_t.reshape(batch, cand_t.shape[-2], cand_t.shape[-1])
    else:
        ref_slices = ref_vals[np.newaxis, ...]
        cand_slices = cand_vals[np.newaxis, ...]

    coord_a = ref.coords[dim_a].values.astype(float)
    coord_b = ref.coords[dim_b].values.astype(float)
    scale_a, scale_b = _pixel_scales(coord_a, coord_b)

    integ_rel = []
    peak_rel = []
    centroid_px = []
    rms_resid = []
    mask_disagree = []

    for rs, cs in zip(ref_slices, cand_slices, strict=True):
        valid_r = np.isfinite(rs)
        valid_c = np.isfinite(cs)
        overlap = valid_r & valid_c
        xor_mask = valid_r ^ valid_c

        frac_xor = float(xor_mask.sum() / xor_mask.size) if xor_mask.size else 0.0
        mask_disagree.append(frac_xor)

        if not overlap.any():
            integ_rel.append(float("nan"))
            peak_rel.append(float("nan"))
            centroid_px.append(float("nan"))
            rms_resid.append(float("nan"))
            continue

        rv = np.where(overlap, rs, np.nan)
        cv = np.where(overlap, cs, np.nan)

        sum_r = float(np.nansum(rv))
        sum_c = float(np.nansum(cv))
        integ_rel.append(abs(sum_c - sum_r) / (abs(sum_r) + eps))

        peak_r = float(np.nanmax(np.abs(rv)))
        peak_c = float(np.nanmax(np.abs(cv)))
        peak_rel.append(abs(peak_c - peak_r) / (abs(peak_r) + eps))

        ca_r, cb_r = _weighted_centroid_2d(rv, coord_a, coord_b, eps=eps)
        ca_c, cb_c = _weighted_centroid_2d(cv, coord_a, coord_b, eps=eps)
        if np.isnan(ca_r) or np.isnan(ca_c) or np.isnan(cb_r) or np.isnan(cb_c):
            centroid_px.append(float("nan"))
        else:
            da_px = (ca_c - ca_r) / scale_a
            db_px = (cb_c - cb_r) / scale_b
            centroid_px.append(float(np.sqrt(da_px**2 + db_px**2)))

        diff = cv - rv
        rms_resid.append(float(np.sqrt(np.nanmean(diff**2))))

    return {
        "integrated_flux_rel_error_mean": float(np.nanmean(integ_rel)),
        "integrated_flux_rel_error_max": float(np.nanmax(integ_rel)),
        "peak_flux_rel_error_mean": float(np.nanmean(peak_rel)),
        "peak_flux_rel_error_max": float(np.nanmax(peak_rel)),
        "centroid_shift_px_mean": float(np.nanmean(centroid_px)),
        "centroid_shift_px_max": float(np.nanmax(centroid_px)),
        "rms_residual_mean": float(np.nanmean(rms_resid)),
        "rms_residual_max": float(np.nanmax(rms_resid)),
        "mask_disagreement_frac_mean": float(np.nanmean(mask_disagree)),
        "mask_disagreement_frac_max": float(np.nanmax(mask_disagree)),
        "per_slice": {
            "integrated_flux_rel_error": _as_list(integ_rel),
            "peak_flux_rel_error": _as_list(peak_rel),
            "centroid_shift_px": _as_list(centroid_px),
            "rms_residual": _as_list(rms_resid),
            "mask_disagreement_frac": _as_list(mask_disagree),
        },
    }


def _evaluate_thresholds(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    checks = {
        "integrated_flux_rel_error_max": metrics["integrated_flux_rel_error_max"]
        <= thresholds["integrated_flux_rel_error_max"],
        "peak_flux_rel_error_max": metrics["peak_flux_rel_error_max"]
        <= thresholds["peak_flux_rel_error_max"],
        "centroid_shift_px_max": metrics["centroid_shift_px_max"]
        <= thresholds["centroid_shift_px_max"],
        "mask_disagreement_frac_max": metrics["mask_disagreement_frac_max"]
        <= thresholds["mask_disagreement_frac_max"],
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _evaluate_thresholds_by_mode(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    quantity_mode: str,
    enable_integrated_check_for_jy_per_beam: bool,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    # For Jy/beam, integrated pixel-sum checks are not scientifically robust by default.
    if quantity_mode != "jy_per_beam" or enable_integrated_check_for_jy_per_beam:
        checks["integrated_flux_rel_error_max"] = (
            metrics["integrated_flux_rel_error_max"]
            <= thresholds["integrated_flux_rel_error_max"]
        )

    checks["peak_flux_rel_error_max"] = (
        metrics["peak_flux_rel_error_max"] <= thresholds["peak_flux_rel_error_max"]
    )
    checks["centroid_shift_px_max"] = (
        metrics["centroid_shift_px_max"] <= thresholds["centroid_shift_px_max"]
    )
    checks["mask_disagreement_frac_max"] = (
        metrics["mask_disagreement_frac_max"] <= thresholds["mask_disagreement_frac_max"]
    )
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run correctness checks for regridding and write JSON report."
    )
    p.add_argument("--input-zarr", type=str, required=True, help="Input Zarr store path.")
    p.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Variable name in Zarr. If omitted, prefers SKY then single data var.",
    )
    p.add_argument(
        "--quantity-mode",
        choices=["auto", "jy_per_pixel", "jy_per_beam", "generic"],
        default="auto",
        help="Physical interpretation of pixel values; auto infers from attrs/units.",
    )
    p.add_argument(
        "--beam-major-arcsec",
        type=float,
        default=None,
        help="Beam major axis FWHM in arcsec (required for quantity-mode=jy_per_beam).",
    )
    p.add_argument(
        "--beam-minor-arcsec",
        type=float,
        default=None,
        help="Beam minor axis FWHM in arcsec (required for quantity-mode=jy_per_beam).",
    )
    p.add_argument(
        "--beam-pa-deg",
        type=float,
        default=None,
        help="Beam position angle in degrees (optional metadata).",
    )
    p.add_argument(
        "--enable-integrated-check-for-jy-per-beam",
        action="store_true",
        help="Also enforce integrated flux check in jy_per_beam mode (disabled by default).",
    )
    p.add_argument("--dim-a", type=str, default=None, help="First spatial dimension.")
    p.add_argument("--dim-b", type=str, default=None, help="Second spatial dimension.")
    p.add_argument(
        "--slice-dim",
        type=str,
        default=None,
        help="Dimension to subsample for faster checks (auto-infer if omitted).",
    )
    p.add_argument("--max-slices", type=int, default=4, help="Maximum slices along slice-dim.")
    p.add_argument(
        "--backend-ref",
        choices=["xarray", "xesmf"],
        default="xarray",
        help="Reference backend.",
    )
    p.add_argument(
        "--backend-cand",
        choices=["xarray", "xesmf"],
        default="xesmf",
        help="Candidate backend to evaluate.",
    )
    p.add_argument("--method", type=str, default="linear", help="Interpolation method.")
    p.add_argument("--n-a-new", type=int, default=None, help="Target size for dim-a.")
    p.add_argument("--n-b-new", type=int, default=None, help="Target size for dim-b.")
    p.add_argument(
        "--fill-value",
        type=float,
        default=np.nan,
        help="Fill value for out-of-domain interpolation points.",
    )
    p.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon.")
    p.add_argument(
        "--thr-integrated-flux-rel-max",
        type=float,
        default=0.01,
        help="Threshold for max relative integrated flux error.",
    )
    p.add_argument(
        "--thr-peak-flux-rel-max",
        type=float,
        default=0.02,
        help="Threshold for max relative peak flux error.",
    )
    p.add_argument(
        "--thr-centroid-shift-px-max",
        type=float,
        default=0.1,
        help="Threshold for max centroid shift in pixels.",
    )
    p.add_argument(
        "--thr-mask-disagreement-max",
        type=float,
        default=0.001,
        help="Threshold for max mask disagreement fraction.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="correctness_report.json",
        help="Output JSON report path.",
    )
    args = p.parse_args()

    ds = xr.open_zarr(args.input_zarr)
    var_name = _select_variable(ds, args.variable)
    da = ds[var_name]

    if args.dim_a is not None and args.dim_b is not None:
        dim_a, dim_b = args.dim_a, args.dim_b
    elif args.dim_a is None and args.dim_b is None:
        dim_a, dim_b = _infer_spatial_dims(da)
    else:
        raise ValueError("Pass both --dim-a and --dim-b, or omit both for auto-detection.")

    if dim_a not in da.dims or dim_b not in da.dims:
        raise ValueError(
            f"Spatial dimensions not found. dims={da.dims}, expected {dim_a!r}/{dim_b!r}"
        )

    quantity_mode = _detect_quantity_mode(ds, da, args.quantity_mode)
    beam_metadata = _resolve_beam_metadata(
        ds=ds,
        da=da,
        quantity_mode=quantity_mode,
        beam_major_arcsec=args.beam_major_arcsec,
        beam_minor_arcsec=args.beam_minor_arcsec,
        beam_pa_deg=args.beam_pa_deg,
    )
    schema_hints = _collect_schema_hints(ds, da)

    slice_dim = args.slice_dim or _infer_slice_dim(da, dim_a, dim_b)
    da_sub = _subset_data(da, slice_dim=slice_dim, max_slices=args.max_slices)
    new_a, new_b = _build_target_coords(
        da_sub,
        dim_a=dim_a,
        dim_b=dim_b,
        n_a_new=args.n_a_new,
        n_b_new=args.n_b_new,
    )

    ref = _run_backend(
        da_sub,
        dim_a=dim_a,
        dim_b=dim_b,
        new_coord_a=new_a,
        new_coord_b=new_b,
        backend=args.backend_ref,
        method=args.method,
        fill_value=args.fill_value,
    ).compute()
    cand = _run_backend(
        da_sub,
        dim_a=dim_a,
        dim_b=dim_b,
        new_coord_a=new_a,
        new_coord_b=new_b,
        backend=args.backend_cand,
        method=args.method,
        fill_value=args.fill_value,
    ).compute()

    metrics = _compute_metrics(ref, cand, dim_a=dim_a, dim_b=dim_b, eps=args.eps)
    thresholds = {
        "integrated_flux_rel_error_max": args.thr_integrated_flux_rel_max,
        "peak_flux_rel_error_max": args.thr_peak_flux_rel_max,
        "centroid_shift_px_max": args.thr_centroid_shift_px_max,
        "mask_disagreement_frac_max": args.thr_mask_disagreement_max,
    }
    verdict = _evaluate_thresholds_by_mode(
        metrics=metrics,
        thresholds=thresholds,
        quantity_mode=quantity_mode,
        enable_integrated_check_for_jy_per_beam=args.enable_integrated_check_for_jy_per_beam,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "input_zarr": args.input_zarr,
            "variable": var_name,
            "quantity_mode": quantity_mode,
            "dim_a": dim_a,
            "dim_b": dim_b,
            "slice_dim": slice_dim,
            "max_slices": args.max_slices,
            "backend_ref": args.backend_ref,
            "backend_cand": args.backend_cand,
            "method": args.method,
            "n_a_new": int(new_a.size),
            "n_b_new": int(new_b.size),
            "fill_value": None if np.isnan(args.fill_value) else args.fill_value,
            "eps": args.eps,
            "integrated_check_enabled": (
                quantity_mode != "jy_per_beam"
                or args.enable_integrated_check_for_jy_per_beam
            ),
            "beam_metadata": beam_metadata,
        },
        "schema_hints": schema_hints,
        "thresholds": thresholds,
        "metrics": metrics,
        "verdict": verdict,
    }

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote correctness report to: {out_path}")
    print(f"Overall pass: {verdict['pass']}")
    for name, passed in verdict["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
