"""
Plot a 2D SKY slice from an XRADIO-style Zarr image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ARCSEC_PER_RAD = 206264.80624709636


def _pick_dim(da: xr.DataArray, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in da.dims:
            return c
    return None


def _isel_if_present(da: xr.DataArray, dim: str | None, idx: int) -> xr.DataArray:
    if dim is None:
        return da
    if idx < 0 or idx >= da.sizes[dim]:
        raise ValueError(f"Index {idx} out of range for dim {dim!r} with size {da.sizes[dim]}")
    return da.isel({dim: idx})


def main() -> None:
    p = argparse.ArgumentParser(description="Plot a 2D slice from an XRADIO Zarr image.")
    p.add_argument("--input-zarr", required=True, help="Path to input .zarr store.")
    p.add_argument("--variable", default="SKY", help="Data variable to plot (default: SKY).")
    p.add_argument("--dim-a", default="l", help="Horizontal spatial dim (default: l).")
    p.add_argument("--dim-b", default="m", help="Vertical spatial dim (default: m).")
    p.add_argument("--time-index", type=int, default=0, help="time index (if present).")
    p.add_argument(
        "--freq-index",
        type=int,
        default=0,
        help="frequency/chan index (if present).",
    )
    p.add_argument(
        "--pol-index",
        type=int,
        default=0,
        help="polarization/pol index (if present).",
    )
    p.add_argument(
        "--percentile-max",
        type=float,
        default=99.5,
        help="Upper percentile for display vmax (default: 99.5).",
    )
    p.add_argument("--cmap", default="inferno", help="Matplotlib colormap.")
    p.add_argument("--output-png", default=None, help="If set, write figure to this PNG path.")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (useful with --output-png).",
    )
    args = p.parse_args()

    ds = xr.open_zarr(args.input_zarr)
    if args.variable not in ds.data_vars:
        raise ValueError(f"Variable {args.variable!r} not found. Available: {list(ds.data_vars)}")

    da = ds[args.variable]
    tdim = _pick_dim(da, ["time"])
    fdim = _pick_dim(da, ["frequency", "chan"])
    pdim = _pick_dim(da, ["polarization", "pol"])

    da = _isel_if_present(da, tdim, args.time_index)
    da = _isel_if_present(da, fdim, args.freq_index)
    da = _isel_if_present(da, pdim, args.pol_index)

    if args.dim_a not in da.dims or args.dim_b not in da.dims:
        raise ValueError(f"Spatial dims {args.dim_a!r}/{args.dim_b!r} not in {da.dims}")

    da2 = da.transpose(args.dim_a, args.dim_b)
    x = ds.coords[args.dim_a].values.astype(float)
    y = ds.coords[args.dim_b].values.astype(float)

    xlabel = args.dim_a
    ylabel = args.dim_b
    if args.dim_a.lower() in {"l", "m", "ra", "dec", "lat", "lon"}:
        x = x * ARCSEC_PER_RAD
        y = y * ARCSEC_PER_RAD
        xlabel = f"{args.dim_a} (arcsec)"
        ylabel = f"{args.dim_b} (arcsec)"

    img = da2.values
    finite = np.isfinite(img)
    vmax = np.nanpercentile(np.abs(img[finite]), args.percentile_max) if finite.any() else 1.0
    vmax = max(float(vmax), 1e-12)
    vmin = -vmax if np.nanmin(img) < 0 else 0.0

    plt.figure(figsize=(6, 5))
    plt.imshow(
        img.T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
    )
    units = da.attrs.get("units", "")
    plt.colorbar(label=str(units))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{Path(args.input_zarr).name}:{args.variable}")
    plt.tight_layout()

    if args.output_png:
        out = Path(args.output_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=180)
        print(f"Wrote {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
