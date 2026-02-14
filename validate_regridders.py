"""
Validate that the 'xarray' and 'xesmf' regridders produce similar results.
"""

from __future__ import annotations

import argparse

import xarray as xr
import numpy as np

from regrid_2d import regrid_2d_planes


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate that xarray and xesmf regridders produce close results."
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
    da = da.isel(level=slice(0, 2), lat=slice(0, 200), lon=slice(0, 400))
    print(f"Using a subset of the data with shape: {da.shape}")

    # Create a non-integer ratio target grid
    new_lat_size = 123
    new_lon_size = 276
    new_lat = np.linspace(
        da.coords["lat"].min().item(), da.coords["lat"].max().item(), new_lat_size
    )
    new_lon = np.linspace(
        da.coords["lon"].min().item(), da.coords["lon"].max().item(), new_lon_size
    )
    print(f"Target grid shape: ({new_lat_size}, {new_lon_size})")

    print("\nRegridding with 'xarray'...")
    da_xarray = regrid_2d_planes(
        da,
        "lat",
        "lon",
        new_lat,
        new_lon,
        regridder_name="xarray",
        method="linear",
    )
    result_xarray = da_xarray.compute()
    print("'xarray' regridding complete.")

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
        xr.testing.assert_allclose(result_xarray, result_xesmf, atol=1e-5, rtol=1e-5)
        print("\n✅ Success: The outputs are numerically close.")
    except AssertionError as e:
        print(f"\n❌ Failure: The outputs are not close.\n{e}")

    # Calculate and print discrepancies regardless of the assertion outcome
    abs_diff = abs(result_xarray - result_xesmf)
    max_abs_diff = abs_diff.max().item()

    # Calculate relative difference, handling potential division by zero
    # Use a small epsilon to avoid division by zero when both are very small
    epsilon = 1e-12
    relative_diff = abs_diff / (abs(result_xarray) + epsilon)
    max_relative_diff = relative_diff.max().item()

    # Find the fractional difference at the location of the largest absolute difference
    max_abs_loc = np.unravel_index(np.argmax(abs_diff.values), abs_diff.shape)
    val_xarray = result_xarray[max_abs_loc].item()
    val_xesmf = result_xesmf[max_abs_loc].item()
    frac_diff_at_max_abs = abs(val_xarray - val_xesmf) / abs(val_xarray) if val_xarray != 0 else np.inf

    print("\nDiscrepancy Analysis:")
    print(f"  Maximum absolute discrepancy: {max_abs_diff:.2e}")
    print(f"  Maximum relative discrepancy: {max_relative_diff:.2e} (can be misleading)")
    print("\nAt location of largest absolute discrepancy:")
    print(f"  xarray value: {val_xarray:.4f}")
    print(f"  xesmf value:  {val_xesmf:.4f}")
    print(f"  Fractional difference: {frac_diff_at_max_abs:.2%}")


if __name__ == "__main__":
    main()
