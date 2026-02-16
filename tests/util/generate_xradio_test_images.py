"""
Generate XRADIO sky-image fixtures using XRADIO image factory methods.

This script uses:
  - xradio.image.make_empty_sky_image
  - xradio.image.write_image

and writes each fixture to a Zarr image store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from xradio.image import make_empty_sky_image, write_image


def _gaussian_2d(
    l_vals: np.ndarray,
    m_vals: np.ndarray,
    amp: float,
    l0: float,
    m0: float,
    sigma_l: float,
    sigma_m: float,
) -> np.ndarray:
    ll, mm = np.meshgrid(l_vals, m_vals, indexing="ij")
    rr = ((ll - l0) / sigma_l) ** 2 + ((mm - m0) / sigma_m) ** 2
    return amp * np.exp(-0.5 * rr)


def _gaussian_2d_rotated(
    l_vals: np.ndarray,
    m_vals: np.ndarray,
    amp: float,
    l0: float,
    m0: float,
    sigma_major: float,
    sigma_minor: float,
    pa_rad: float,
) -> np.ndarray:
    ll, mm = np.meshgrid(l_vals, m_vals, indexing="ij")
    dl = ll - l0
    dm = mm - m0
    c = np.cos(pa_rad)
    s = np.sin(pa_rad)
    u = c * dl + s * dm
    v = -s * dl + c * dm
    rr = (u / sigma_major) ** 2 + (v / sigma_minor) ** 2
    return amp * np.exp(-0.5 * rr)


def _build_template(
    n_l: int, n_m: int, n_chan: int, n_pol: int, n_time: int, fov_arcsec: float
) -> xr.Dataset:
    phase_center = [0.0, 0.0]  # [ra, dec] in rad (small-field test frame)
    image_size = [n_l, n_m]
    # Cell size in radians
    cell = np.deg2rad((fov_arcsec / max(n_l, n_m)) / 3600.0)
    cell_size = [cell, cell]
    frequency_coords = np.linspace(1.4e9, 1.401e9, n_chan)
    pol_coords = ["I"] if n_pol == 1 else ["I", "Q", "U", "V"][:n_pol]
    time_coords = np.linspace(59000.0, 59000.0 + (n_time - 1) * 1e-3, n_time)

    xds = make_empty_sky_image(
        phase_center=phase_center,
        image_size=image_size,
        cell_size=cell_size,
        frequency_coords=frequency_coords,
        pol_coords=pol_coords,
        time_coords=time_coords,
        direction_reference="fk5",
        projection="SIN",
        spectral_reference="lsrk",
        do_sky_coords=True,
    )
    return xds


def _broadcast_to_sky(base_2d: np.ndarray, xds: xr.Dataset) -> np.ndarray:
    n_time = xds.sizes["time"]
    n_freq = xds.sizes["frequency"]
    n_pol = xds.sizes["polarization"]
    n_l = xds.sizes["l"]
    n_m = xds.sizes["m"]
    return np.broadcast_to(base_2d[None, None, None, :, :], (n_time, n_freq, n_pol, n_l, n_m)).copy()


def _attach_sky_and_flag(
    xds: xr.Dataset,
    sky_vals: np.ndarray,
    units: str,
    beam_major_arcsec: float | None = None,
    beam_minor_arcsec: float | None = None,
    beam_pa_deg: float | None = None,
) -> xr.Dataset:
    dims = ("time", "frequency", "polarization", "l", "m")
    coords = {d: xds.coords[d] for d in dims}
    xds["SKY"] = xr.DataArray(sky_vals, dims=dims, coords=coords)
    xds["SKY"].attrs.update({"image_type": "Intensity", "type": "sky", "units": units})
    flag = np.zeros_like(sky_vals, dtype=bool)
    # In xradio FLAG_SKY, True means masked/invalid
    flag[..., : max(2, xds.sizes["l"] // 20), : max(2, xds.sizes["m"] // 20)] = True
    xds["FLAG_SKY"] = xr.DataArray(flag, dims=dims, coords=coords)
    xds["FLAG_SKY"].attrs.update({"type": "mask"})
    xds["SKY"] = xds["SKY"].where(~xds["FLAG_SKY"], np.nan)

    xds.attrs["data_groups"] = {"base": {"sky": "SKY", "flag": "FLAG_SKY"}}
    xds.attrs["type"] = "image_dataset"

    if (
        beam_major_arcsec is not None
        and beam_minor_arcsec is not None
        and beam_pa_deg is not None
    ):
        # XRADIO schema: one beam triplet per (time, frequency, polarization),
        # encoded as BEAM_FIT_PARAMS[..., beam_params_label=["major","minor","pa"]].
        # Use a single angular unit on beam_params_label as requested.
        beam_labels = np.array(["major", "minor", "pa"], dtype=object)
        beam_vals = np.array(
            [
                np.deg2rad(float(beam_major_arcsec) / 3600.0),
                np.deg2rad(float(beam_minor_arcsec) / 3600.0),
                np.deg2rad(float(beam_pa_deg)),
            ],
            dtype=np.float64,
        )
        beam = np.broadcast_to(
            beam_vals[None, None, None, :],
            (
                xds.sizes["time"],
                xds.sizes["frequency"],
                xds.sizes["polarization"],
                3,
            ),
        ).copy()
        xds["BEAM_FIT_PARAMS"] = xr.DataArray(
            beam,
            dims=("time", "frequency", "polarization", "beam_params_label"),
            coords={
                "time": xds.coords["time"],
                "frequency": xds.coords["frequency"],
                "polarization": xds.coords["polarization"],
                "beam_params_label": beam_labels,
            },
        )
        xds["beam_params_label"].attrs["units"] = "rad"
        xds["BEAM_FIT_PARAMS"].attrs["units"] = "rad"
    return xds


def _case_point_source_jy_per_beam(xds: xr.Dataset) -> tuple[xr.Dataset, str]:
    l = xds.coords["l"].values
    m = xds.coords["m"].values
    # Keep beam broader than pixel scale so the source is visibly beam-shaped.
    beam_major_arcsec = 30.0
    beam_minor_arcsec = 24.0
    beam_pa_deg = 35.0
    fwhm_major_rad = np.deg2rad(beam_major_arcsec / 3600.0)
    fwhm_minor_rad = np.deg2rad(beam_minor_arcsec / 3600.0)
    sigma_major = fwhm_major_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_minor = fwhm_minor_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    base = _gaussian_2d_rotated(
        l,
        m,
        amp=1.0,
        l0=l[l.size // 2],
        m0=m[m.size // 2],
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        pa_rad=np.deg2rad(beam_pa_deg),
    )
    sky = _broadcast_to_sky(base, xds)
    return _attach_sky_and_flag(
        xds,
        sky,
        units="Jy/beam",
        beam_major_arcsec=beam_major_arcsec,
        beam_minor_arcsec=beam_minor_arcsec,
        beam_pa_deg=beam_pa_deg,
    ), "jy_per_beam"


def _case_point_source_jy_per_pixel(xds: xr.Dataset) -> tuple[xr.Dataset, str]:
    l = xds.coords["l"].values
    m = xds.coords["m"].values
    base = np.zeros((l.size, m.size), dtype=np.float64)
    base[l.size // 2, m.size // 2] = 1.0
    sky = _broadcast_to_sky(base, xds)
    return _attach_sky_and_flag(xds, sky, units="Jy/pixel"), "jy_per_pixel"


def _case_two_source_blend(xds: xr.Dataset) -> tuple[xr.Dataset, str]:
    l = xds.coords["l"].values
    m = xds.coords["m"].values
    g1 = _gaussian_2d(
        l, m, amp=1.0, l0=l[l.size // 3], m0=m[m.size // 3], sigma_l=3e-5, sigma_m=2e-5
    )
    g2 = _gaussian_2d(
        l,
        m,
        amp=0.65,
        l0=l[(2 * l.size) // 3],
        m0=m[(2 * m.size) // 3],
        sigma_l=2e-5,
        sigma_m=3e-5,
    )
    sky = _broadcast_to_sky(g1 + g2, xds)
    return _attach_sky_and_flag(
        xds,
        sky,
        units="Jy/beam",
        beam_major_arcsec=1.9,
        beam_minor_arcsec=1.4,
        beam_pa_deg=10.0,
    ), "jy_per_beam"


def _case_extended_gaussian(xds: xr.Dataset) -> tuple[xr.Dataset, str]:
    l = xds.coords["l"].values
    m = xds.coords["m"].values
    base = _gaussian_2d(
        l,
        m,
        amp=3.0,
        l0=0.0,
        m0=0.0,
        sigma_l=(l.max() - l.min()) / 8.0,
        sigma_m=(m.max() - m.min()) / 10.0,
    )
    sky = _broadcast_to_sky(base, xds)
    return _attach_sky_and_flag(xds, sky, units="Jy/pixel"), "jy_per_pixel"


def _case_flat_gradient(xds: xr.Dataset) -> tuple[xr.Dataset, str]:
    l = xds.coords["l"].values
    m = xds.coords["m"].values
    ll, mm = np.meshgrid(l, m, indexing="ij")
    base = 0.15 + 0.02 * (ll / np.max(np.abs(l))) + 0.01 * (mm / np.max(np.abs(m)))
    sky = _broadcast_to_sky(base, xds)
    return _attach_sky_and_flag(xds, sky, units="Jy/pixel"), "jy_per_pixel"


def _case_edge_structure(xds: xr.Dataset) -> tuple[xr.Dataset, str]:
    l = xds.coords["l"].values
    m = xds.coords["m"].values
    base = np.zeros((l.size, m.size), dtype=np.float64)
    base += _gaussian_2d(
        l,
        m,
        amp=1.3,
        l0=l[int(0.92 * l.size)],
        m0=m[int(0.08 * m.size)],
        sigma_l=2.5e-5,
        sigma_m=2.5e-5,
    )
    sky = _broadcast_to_sky(base, xds)
    return _attach_sky_and_flag(
        xds,
        sky,
        units="Jy/beam",
        beam_major_arcsec=2.3,
        beam_minor_arcsec=1.8,
        beam_pa_deg=75.0,
    ), "jy_per_beam"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate XRADIO image fixtures using xradio image factory methods."
    )
    p.add_argument("--output-dir", type=str, required=True, help="Directory for output Zarr images.")
    p.add_argument("--n-l", type=int, default=128, help="Pixels along l.")
    p.add_argument("--n-m", type=int, default=128, help="Pixels along m.")
    p.add_argument("--n-chan", type=int, default=4, help="Number of frequency channels.")
    p.add_argument("--n-pol", type=int, default=1, help="Number of polarization planes.")
    p.add_argument("--n-time", type=int, default=1, help="Number of time planes.")
    p.add_argument("--fov-arcsec", type=float, default=1200.0, help="Field of view (arcsec).")
    p.add_argument(
        "--cases",
        nargs="+",
        default=[
            "point_source_center_jy_per_beam",
            "point_source_center_jy_per_pixel",
            "two_source_blend_jy_per_beam",
            "extended_gaussian_jy_per_pixel",
            "flat_field_gradient_jy_per_pixel",
            "edge_structure_jy_per_beam",
        ],
        help="Cases to generate.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "point_source_center_jy_per_beam": _case_point_source_jy_per_beam,
        "point_source_center_jy_per_pixel": _case_point_source_jy_per_pixel,
        "two_source_blend_jy_per_beam": _case_two_source_blend,
        "extended_gaussian_jy_per_pixel": _case_extended_gaussian,
        "flat_field_gradient_jy_per_pixel": _case_flat_gradient,
        "edge_structure_jy_per_beam": _case_edge_structure,
    }
    unknown = [c for c in args.cases if c not in builders]
    if unknown:
        raise ValueError(f"Unknown case names: {unknown}. Valid values: {sorted(builders)}")

    manifest: dict[str, dict[str, str]] = {}
    for case in args.cases:
        xds = _build_template(
            n_l=args.n_l,
            n_m=args.n_m,
            n_chan=args.n_chan,
            n_pol=args.n_pol,
            n_time=args.n_time,
            fov_arcsec=args.fov_arcsec,
        )
        xds_case, mode = builders[case](xds)
        out_path = out_dir / f"{case}.zarr"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Use --overwrite to replace.")
        write_image(xds_case, str(out_path), out_format="zarr", overwrite=args.overwrite)
        manifest[case] = {
            "path": str(out_path),
            "variable": "SKY",
            "dim_a": "l",
            "dim_b": "m",
            "quantity_mode_suggestion": mode,
            "units": str(xds_case["SKY"].attrs.get("units", "unknown")),
        }
        print(f"Wrote {case}: {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
