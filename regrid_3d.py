"""
Regrid a 3D xarray.DataArray along two dimensions in parallel over the third.

Concept:
- DataArray has dims like (dim_independent, dim_a, dim_b), e.g. (level, lat, lon).
- We regrid (dim_a, dim_b) onto a new 2D grid.
- Each slice along dim_independent is regridded independently, so we use Dask
  to parallelize over that dimension.

Usage:
    from dask_regrid.regrid_3d import regrid_2d_planes, make_example_3d

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
from scipy.interpolate import RegularGridInterpolator


def _regrid_single_plane(
    plane: np.ndarray,
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    points: np.ndarray,
    new_shape: tuple[int, int],
    method: str = "linear",
    fill_value: float | None = np.nan,
) -> np.ndarray:
    """
    Regrid a single 2D plane onto new coordinates using scipy.

    This version uses a pre-computed grid of points for the new coordinates
    to avoid repeated calculations.

    Parameters
    ----------
    plane : (n_a, n_b)
        2D array to regrid.
    coord_a, coord_b : 1D arrays
        Current grid coordinates (e.g. lat, lon).
    points : (n_new_a * n_new_b, 2)
        Pre-computed grid of points for the new coordinates.
    new_shape : (n_new_a, n_new_b)
        Shape of the new grid.
    method : str
        Passed to RegularGridInterpolator: 'linear', 'nearest', etc.
    fill_value : float or None
        Value for points outside the original grid.

    Returns
    -------
    out : (n_new_a, n_new_b)
        Regridded 2D array.
    """
    interpolator = RegularGridInterpolator(
        (coord_a, coord_b),
        plane,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )
    out_flat = interpolator(points)
    return out_flat.reshape(new_shape)


def _regrid_block_func(
    block: np.ndarray,
    *,
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    points: np.ndarray,
    new_shape: tuple[int, int],
    method: str,
    fill_value: float | None,
) -> np.ndarray:
    # block shape: (n_independent, n_a, n_b)
    n_ind = block.shape[0]
    out = np.empty(
        (n_ind, new_shape[0], new_shape[1]),
        dtype=block.dtype,
    )
    for i in range(n_ind):
        out[i] = _regrid_single_plane(
            block[i],
            coord_a,
            coord_b,
            points,
            new_shape,
            method=method,
            fill_value=fill_value,
        )
    return out


import xesmf as xe
...
def regrid_2d_planes(
    da: xr.DataArray,
    dim_a: str,
    dim_b: str,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    regridder_name: str = "scipy",
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
    regridder_name : {'scipy', 'xesmf'}
        Name of the regridding backend to use.
    method : str
        Interpolation method: "linear", "nearest", etc.
    fill_value : float or None
        Value for points outside the source grid.
    chunk_size : int or None
        Chunk size along the independent dimension. If None, use one chunk
        per slice (maximum parallelism, more overhead) or keep existing chunks.

    Returns
    -------
    xr.DataArray
        Same as da but with dim_a and dim_b replaced by the new grids.
        If da was lazy (Dask-backed), the result is lazy and can be
        computed with .compute().
    """
    if regridder_name == "scipy":
        return _regrid_2d_planes_scipy(
            da, dim_a, dim_b, new_coord_a, new_coord_b, method, fill_value, chunk_size
        )
    elif regridder_name == "xesmf":
        return _regrid_2d_planes_xesmf(da, new_coord_a, new_coord_b, method)
    else:
        raise ValueError(f"Unknown regridder_name: {regridder_name!r}")


def _regrid_2d_planes_xesmf(
    da: xr.DataArray, new_coord_a: np.ndarray, new_coord_b: np.ndarray, method: str
) -> xr.DataArray:
    """Regrid using xESMF."""
    if method == "linear":
        method = "bilinear"  # xESMF uses 'bilinear' for linear interpolation
    ds_out = xr.Dataset(coords={"lat": new_coord_a, "lon": new_coord_b})
    regridder = xe.Regridder(da, ds_out, method=method)
    return regridder(da)

def _regrid_2d_planes_scipy(
    da: xr.DataArray,
    dim_a: str,
    dim_b: str,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    method: str = "linear",
    fill_value: float | None = np.nan,
    chunk_size: int | None = None,
) -> xr.DataArray:
    """The original regridding implementation using SciPy."""
    dim_independent = [d for d in da.dims if d not in (dim_a, dim_b)]
    if len(dim_independent) != 1:
        raise ValueError(
            f"DataArray must have exactly three dimensions. "
            f"Found dims: {list(da.dims)}; dim_a={dim_a}, dim_b={dim_b}."
        )
    dim_independent = dim_independent[0]

    coord_a = np.asarray(da.coords[dim_a].values)
    coord_b = np.asarray(da.coords[dim_b].values)
    new_coord_a = np.asarray(new_coord_a)
    new_coord_b = np.asarray(new_coord_b)

    # Pre-compute the grid of points for the new coordinates. This is a
    # major optimization because this grid is the same for all planes.
    new_a_2d, new_b_2d = np.meshgrid(new_coord_a, new_coord_b, indexing="ij")
    points = np.stack([new_a_2d.ravel(), new_b_2d.ravel()], axis=1)
    new_shape = (len(new_coord_a), len(new_coord_b))

    # Ensure we have Dask arrays and chunk along the independent dimension
    if not hasattr(da.data, "rechunk"):
        da = da.chunk({dim_independent: chunk_size or 1})
    elif chunk_size is not None:
        da = da.chunk({dim_independent: chunk_size})

    # Output chunk shape: (chunk_independent, len(new_coord_a), len(new_coord_b))
    out_chunks = (
        da.chunks[da.dims.index(dim_independent)],
        (len(new_coord_a),),
        (len(new_coord_b),),
    )
    result_data = da.data.map_blocks(
        _regrid_block_func,
        coord_a=coord_a,
        coord_b=coord_b,
        points=points,
        new_shape=new_shape,
        method=method,
        fill_value=fill_value,
        dtype=da.dtype,
        chunks=out_chunks,
        meta=np.array((), dtype=da.dtype),
    )

    # Build output coordinates: same independent dim, new dim_a and dim_b
    new_coords = {
        dim_independent: da.coords[dim_independent],
        dim_a: new_coord_a,
        dim_b: new_coord_b,
    }

    return xr.DataArray(
        result_data,
        dims=[dim_independent, dim_a, dim_b],
        coords=new_coords,
        attrs=da.attrs,
        name=da.name,
    )


import dask.array as da
...
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
    )
    print(f"  Output shape: {da_new.shape}")

    result = da_new.compute()
    print(f"  After .compute(): shape {result.shape}, dtype {result.dtype}")
    print("\nDone. Use a Dask scheduler (e.g. distributed) to see parallel execution.")


if __name__ == "__main__":
    main()
