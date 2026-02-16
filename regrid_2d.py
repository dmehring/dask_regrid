"""
Regrid an n-dimensional xarray.DataArray along two named spatial dimensions.

Concept:
- DataArray can have any number of dimensions >= 2.
- We regrid (dim_a, dim_b) onto a new 2D grid.
- All other dimensions are preserved.
- If the DataArray is Dask-backed, parallelism is controlled by chunking and the
  active Dask scheduler across dimensions other than `dim_a` and `dim_b`.

Usage:
    from regrid_2d import regrid_2d_planes, make_example_3d

    xda = make_example_3d()
    da_new = regrid_2d_planes(
        xda,
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
    xda: xr.DataArray,
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
    Regrid an n-dimensional DataArray along two named dimensions.

    The dimensions `dim_a` and `dim_b` are regridded. Any remaining dimensions
    (e.g. time, frequency, polarization) are preserved and broadcast over.
    For Dask-backed arrays, parallel execution is determined by chunking and the
    chosen Dask scheduler across dimensions other than `dim_a` and `dim_b`.

    Parameters
    ----------
    xda : xr.DataArray
        Input array with dimensions that include `dim_a` and `dim_b`.
    dim_a, dim_b : str
        Names of the two dimensions to regrid (e.g. "lat", "lon").
    new_coord_a, new_coord_b : array-like
        1D coordinate values for the output grid along `dim_a` and `dim_b`.
        These vectors define where samples are evaluated in the regridded
        result, i.e. the new axis coordinates used by interpolation/regridding.
    regridder_name : {'xarray', 'xesmf'}
        Name of the regridding backend to use.
    method : str
        Interpolation method: "linear", "nearest", etc.
    fill_value : float or None
        Value for points outside the source grid. (Only used for 'xarray' backend).
    chunk_size : int | None
        Reserved for future chunk-control behavior. Currently unused.

    Returns
    -------
    xr.DataArray
        Same as `xda` but with dim_a and dim_b replaced by the new grids.
        If `xda` was lazy (Dask-backed), the result is lazy and can be
        computed with .compute().
    """
    if regridder_name == "xarray":
        return _regrid_2d_planes_xarray(
            xda, dim_a, dim_b, new_coord_a, new_coord_b, method, fill_value
        )
    elif regridder_name == "xesmf":
        return _regrid_2d_planes_xesmf(
            xda,
            new_coord_a,
            new_coord_b,
            method,
            dim_a=dim_a,
            dim_b=dim_b,
        )
    else:
        raise ValueError(f"Unknown regridder_name: {regridder_name!r}")


def _regrid_2d_planes_xesmf(
    xda: xr.DataArray,
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

    da_in = xda.rename(rename_map) if rename_map else xda
    ds_out = xr.Dataset(coords={"lat": new_coord_a, "lon": new_coord_b})
    regridder = xe.Regridder(da_in, ds_out, method=method)
    out = regridder(da_in)

    if rename_map:
        inv_rename_map = {v: k for k, v in rename_map.items()}
        out = out.rename(inv_rename_map)
    return out


def _regrid_2d_planes_xarray(
    xda: xr.DataArray,
    dim_a: str,
    dim_b: str,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    method: str,
    fill_value: float | None,
) -> xr.DataArray:
    """Regridding implementation using xarray.interp."""
    return xda.interp(
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
    xda = make_example_3d(n_level=4, n_lat=90, n_lon=180, chunk_level=1)
    print(f"  Shape: {xda.shape}, dims: {xda.dims}")
    print(f"  Chunks: {xda.chunks}")

    # Regrid to a coarser lat/lon grid
    new_lat = np.linspace(-90, 90, 45)
    new_lon = np.linspace(0, 360, 90)
    print("\nRegridding to lat=45, lon=90 (parallel over level)...")
    da_new = regrid_2d_planes(
        xda,
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
