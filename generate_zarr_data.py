"""
Generate a large 3D xarray.DataArray with Dask backing and save to a Zarr store.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from regrid_3d import make_example_3d


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a sample Dask-backed xarray.DataArray and save to Zarr."
    )
    p.add_argument(
        "--levels",
        type=int,
        default=256,
        help="Number of levels (independent dimension)",
    )
    p.add_argument(
        "--lat",
        type=int,
        default=2000,
        help="Source grid latitude points",
    )
    p.add_argument(
        "--lon",
        type=int,
        default=4000,
        help="Source grid longitude points",
    )
    p.add_argument(
        "--chunk-level",
        type=int,
        default=8,
        help="Chunk size along level dimension",
    )
    p.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the Zarr store (e.g. /tmp/test_data.zarr)",
    )
    args = p.parse_args()

    print("Generating Dask-backed xarray.DataArray...")
    print(f"  Shape: ({args.levels}, {args.lat}, {args.lon})")
    print(f"  Chunking: level={args.chunk_level}")
    da = make_example_3d(
        n_level=args.levels,
        n_lat=args.lat,
        n_lon=args.lon,
        chunk_level=args.chunk_level,
    )

    print(f"\nSaving to Zarr store at: {args.output_path}")
    da.to_zarr(args.output_path, mode="w")

    print("\nDone.")


if __name__ == "__main__":
    main()
