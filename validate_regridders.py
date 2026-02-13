"""
Validate that the 'scipy' and 'xesmf' regridders produce similar results.
"""

from __future__ import annotations

import argparse

import xarray as xr

from regrid_3d import regrid_2d_planes


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate that scipy and xesmf regridders produce close results."
    )
    p.add_argument(
        "--input-zarr",
        type=str,
        required=True,
        help="Path to input Zarr store.",
    )
    p.add_argument(
        "--lat-new",
        type=int,
        default=1000,
        help="Target grid latitude points.",
    )
    p.add_argument(
        "--lon-new",
        type=int,
        default=2000,
        help="Target grid longitude points.",
    )
    args = p.parse_args()

    print(f"Loading data from {args.input_zarr}...")
    da = xr.open_zarr(args.input_zarr)["temperature"]

    # Use a smaller subset for faster validation
    da = da.isel(level=slice(0, 2))
    print(f"Using a subset of the data with shape: {da.shape}")

    new_lat = da.coords["lat"].values[: args.lat_new]
    new_lon = da.coords["lon"].values[: args.lon_new]

    print("\nRegridding with 'scipy'...")
    da_scipy = regrid_2d_planes(
        da,
        "lat",
        "lon",
        new_lat,
        new_lon,
        regridder_name="scipy",
        method="linear",
    )
    result_scipy = da_scipy.compute()
    print("'scipy' regridding complete.")

    print("\nRegridding with 'xesmf'...")
    da_xesmf = regrid_2d_planes(
        da,
        "lat",
        "lon",
        new_lat,
        new_lon,
        regridder_name="xesmf",
        method="linear",
    )
    result_xesmf = da_xesmf.compute()
    print("'xesmf' regridding complete.")

    print("\nComparing results...")
    try:
        xr.testing.assert_allclose(result_scipy, result_xesmf, atol=1e-5, rtol=1e-5)
        print("\n✅ Success: The outputs are numerically close.")

        abs_diff = abs(result_scipy - result_xesmf)
        max_abs_diff = abs_diff.max().item()

        # Calculate relative difference, handling potential division by zero
        # Use a small epsilon to avoid division by zero when both are very small
        epsilon = 1e-12
        relative_diff = abs_diff / (abs(result_scipy) + epsilon)
        max_relative_diff = relative_diff.max().item()

        print(f"  Maximum absolute discrepancy: {max_abs_diff:.2e}")
        print(f"  Maximum relative discrepancy: {max_relative_diff:.2e}")

    except AssertionError as e:
        print(f"\n❌ Failure: The outputs are not close.\n{e}")


if __name__ == "__main__":
    main()
