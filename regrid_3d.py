"""
Regrid a 3D xarray.DataArray along two dimensions in parallel over the third.

Concept:
- DataArray has dims like (dim_independent, dim_a, dim_b), e.g. (level, lat, lon).
- We regrid (dim_a, dim_b) onto a new 2D grid.
- Each slice along dim_independent is regridded independently, so we use Dask
  to parallelize over that dimension.

Usage:
    from regrid_3d import regrid_2d_planes, make_example_3d

    da = make_example_3d()
    da_new = regrid_2d_planes(
        da,
        dim_a="lat", dim_b="lon",
        new_coord_a=np.linspace(-90, 90, 180),
        new_coord_b=np.linspace(0, 360, 360),
        method="linear",
    )
    result = da_new.compute()
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import xesmf as xe
import dask.array as da


def regrid_2d_planes(
    da: xr.DataArray,
    dim_a: str,
    dim_b: str,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    regridder_name: str = "xarray",
    method: str = "linear",
    fill_value: float | None = np.nan,
    chunk_size: int | None = None,
) -> xr.DataArray:
    """
    Regrid a 3D DataArray along two dimensions, in parallel over the third.

    The dimension that is not dim_a or dim_b is the "independent" dimension
    (e.g. level, time). Each slice along that dimension is regridded
    independently; Dask chunks along that dimension and runs the regrid in
    parallel.

    Parameters
    ----------
    da : xr.DataArray
        Must have exactly three dimensions: dim_independent, dim_a, dim_b.
    dim_a, dim_b : str
        Names of the two dimensions to regrid (e.g. "lat", "lon").
    new_coord_a, new_coord_b : array-like
        1D arrays of target coordinates.
    regridder_name : {'xarray', 'xesmf'}
        Name of the regridding backend to use.
    method : str
        Interpolation method: "linear", "nearest", etc.
    fill_value : float or None
        Value for points outside the source grid. (Only used for 'xarray' backend).
    chunk_size : int | None
        Chunk size along the independent dimension. (Only used for 'xarray' backend).

    Returns
    -------
    xr.DataArray
        Same as da but with dim_a and dim_b replaced by the new grids.
        If da was lazy (Dask-backed), the result is lazy and can be
        computed with .compute().
    """
    if regridder_name == "xarray":
        return _regrid_2d_planes_xarray(
            da, dim_a, dim_b, new_coord_a, new_coord_b, method, fill_value
        )
    elif regridder_name == "xesmf":
        return _regrid_2d_planes_xesmf(
            da,
            new_coord_a,
            new_coord_b,
            method,
            dim_a=dim_a,
            dim_b=dim_b,
        )
    else:
        raise ValueError(f"Unknown regridder_name: {regridder_name!r}")


def _regrid_2d_planes_xesmf(
    da: xr.DataArray,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    method: str,
    dim_a: str = "lat",
    dim_b: str = "lon",
) -> xr.DataArray:
    """Regrid using xESMF."""
    if method == "linear":
        method = "bilinear"  # xESMF uses 'bilinear' for linear interpolation

    # xESMF expects canonical horizontal names ('lat', 'lon'). Keep this internal
    # so callers can use domain-specific names like (l, m).
    rename_map: dict[str, str] = {}
    if dim_a != "lat":
        rename_map[dim_a] = "lat"
    if dim_b != "lon":
        rename_map[dim_b] = "lon"

    da_in = da.rename(rename_map) if rename_map else da
    ds_out = xr.Dataset(coords={"lat": new_coord_a, "lon": new_coord_b})
    regridder = xe.Regridder(da_in, ds_out, method=method)
    out = regridder(da_in)

    if rename_map:
        inv_rename_map = {v: k for k, v in rename_map.items()}
        out = out.rename(inv_rename_map)
    return out


def _regrid_2d_planes_xarray(
    da: xr.DataArray,
    dim_a: str,
    dim_b: str,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    method: str,
    fill_value: float | None,
) -> xr.DataArray:
    """Regridding implementation using xarray.interp."""
    return da.interp(
        coords={dim_a: new_coord_a, dim_b: new_coord_b},
        method=method,
        kwargs={"fill_value": fill_value},
    )


def make_example_3d(
    n_level: int = 4,
    n_lat: int = 90,
    n_lon: int = 180,
    chunk_level: int = 1,
    seed: int | None = 42,
) -> xr.DataArray:
    """
    Create a small 3D example DataArray (level, lat, lon) with Dask chunks.

    Useful to try regrid_2d_planes without loading real data.
    """
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(0, 360, n_lon)
    level = np.arange(n_level)

    # Synthetic data: (level, lat, lon)
    rng = da.random.RandomState(seed)
    data = rng.normal(
        0, 1, (n_level, n_lat, n_lon), chunks=(chunk_level, n_lat, n_lon)
    ).astype(np.float64)

    da_xr = xr.DataArray(
        data,
        dims=["level", "lat", "lon"],
        coords={"level": level, "lat": lat, "lon": lon},
        name="temperature",
    )
    return da_xr


def main() -> None:
    """Run a small example: create 3D data, regrid, and print shapes."""
    print("Creating example 3D DataArray (level=4, lat=90, lon=180)...")
    da = make_example_3d(n_level=4, n_lat=90, n_lon=180, chunk_level=1)
    print(f"  Shape: {da.shape}, dims: {da.dims}")
    print(f"  Chunks: {da.chunks}")

    # Regrid to a coarser lat/lon grid
    new_lat = np.linspace(-90, 90, 45)
    new_lon = np.linspace(0, 360, 90)
    print("\nRegridding to lat=45, lon=90 (parallel over level)...")
    da_new = regrid_2d_planes(
        da,
        dim_a="lat",
        dim_b="lon",
        new_coord_a=new_lat,
        new_coord_b=new_lon,
        method="linear",
        regridder_name="xarray",
    )
    print(f"  Output shape: {da_new.shape}")

    result = da_new.compute()
    print(f"  After .compute(): shape {result.shape}, dtype {result.dtype}")
    print("\nDone. Use a Dask scheduler (e.g. distributed) to see parallel execution.")


if __name__ == "__main__":
    main()
