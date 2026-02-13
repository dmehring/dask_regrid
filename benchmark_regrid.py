"""
Benchmark regrid under different Dask schedulers on a large 3D xarray.DataArray.

Schedulers:
  - sync (synchronous): single-threaded. Often fastest for in-memory data on one
    machine, since there is no serialization or IPC.
  - threads: multi-threaded. Can be limited by the GIL for CPU-bound work.
  - processes: multi-process. True parallelism but each chunk is serialized and
    sent to workers; for large in-memory arrays this overhead usually dominates,
    so processes can be slower than sync on a single machine.
  - distributed: More advanced scheduler that runs on a local cluster.

Use --scheduler to run one, or "all" (default) to compare all three.

To see threads clearly win, use:  --demo-threads
That preset uses: enough levels to keep workers busy, moderate 2D size so each
slice is CPU-bound (SciPy releases GIL), and chunk_level=1 so process
serialization cost is high. Overrides --levels, --lat, --lon, --lat-new, --lon-new.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from regrid_3d import make_example_3d, regrid_2d_planes


def _time_compute(
    da: xr.DataArray,
    new_coord_a: np.ndarray,
    new_coord_b: np.ndarray,
    regridder_name: str,
    scheduler: str,
    n_workers: int | None,
) -> tuple[float, object]:
    """Run compute with the given scheduler; return (seconds, result)."""
    import dask

    print(f"  Regridding with '{regridder_name}' backend...")
    da_new = regrid_2d_planes(
        da,
        dim_a="lat",
        dim_b="lon",
        new_coord_a=new_coord_a,
        new_coord_b=new_coord_b,
        regridder_name=regridder_name,
        method="linear",
    )

    if scheduler == "distributed":
        from dask.distributed import Client, LocalCluster

        with LocalCluster(
            n_workers=1, threads_per_worker=n_workers or 1, processes=True
        ) as cluster:
            with Client(cluster) as client:
                print(f"  Starting compute on {client.dashboard_link} ...")
                t0 = time.perf_counter()
                result = da_new.compute()
                return time.perf_counter() - t0, result
    else:
        # For sync, threads, processes
        config = {"scheduler": scheduler}
        if scheduler != "sync":
            config["num_workers"] = n_workers
        with dask.config.set(config):
            print("  Starting compute...")
            t0 = time.perf_counter()
            result = da_new.compute()
            return time.perf_counter() - t0, result


def run_benchmark(
    input_zarr: str | None,
    n_level: int,
    n_lat: int,
    n_lon: int,
    n_lat_new: int,
    n_lon_new: int,
    chunk_level: int,
    n_workers: int,
    schedulers: list[str],
    regridder_name: str,
) -> None:
    """Create a large 3D DataArray, run regrid under selected schedulers, report timings."""
    if input_zarr:
        print(f"Loading xarray.DataArray from Zarr store: {input_zarr}")
        da = xr.open_zarr(input_zarr)["temperature"]
        print(f"  Shape: {da.shape} = {da.size:,} points")
        print(f"  Chunks: {da.chunks}")
    else:
        print("Building large 3D xarray.DataArray (level, lat, lon)...")
        print(
            f"  Shape: ({n_level}, {n_lat}, {n_lon}) = {n_level * n_lat * n_lon:,} points"
        )
        print(f"  Chunking: level={chunk_level} (one task per level slice)")
        da = make_example_3d(
            n_level=n_level,
            n_lat=n_lat,
            n_lon=n_lon,
            chunk_level=chunk_level,
            seed=42,
        )

    print(f"Target grid: ({n_lat_new}, {n_lon_new})")
    print()

    new_lat = np.linspace(-90, 90, n_lat_new)
    new_lon = np.linspace(0, 360, n_lon_new)

    times: dict[str, float] = {}
    for sched in schedulers:
        label = f"{sched} (n_workers={n_workers})" if sched != "sync" else "sync"
        print(f"Scheduler: {label}...")
        t, result = _time_compute(
            da,
            new_lat,
            new_lon,
            regridder_name,
            sched,
            n_workers if sched != "sync" else None,
        )
        times[sched] = t
        print(f"  Time: {t:.2f} s  |  Result: {type(result).__name__} {result.shape}")
    print()

    # Summary table
    t_baseline = times.get("sync") or times.get(schedulers[0])
    print("--- Summary ---")
    for sched in schedulers:
        t = times[sched]
        speedup = t_baseline / t if t > 0 else float("inf")
        print(f"  {sched:12s}  {t:6.2f} s   (vs sync: {speedup:.2f}x)")
    print()
    if "processes" in times and times.get("processes", 0) > times.get(
        "sync", float("inf")
    ):
        print(
            "Note: Process scheduler is slower here because each chunk is serialized "
            "and sent to workers. For one big in-memory array on a single machine, "
            "synchronous is usually fastest."
        )
    print("Done. Results are xarray.DataArrays (not raw numpy).")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark regrid: single-threaded vs multi-process on a large 3D DataArray."
    )
    p.add_argument(
        "--input-zarr",
        type=str,
        default=None,
        help="Path to input Zarr store. If not provided, data is generated in memory.",
    )
    p.add_argument(
        "--regridder",
        choices=["scipy", "xesmf"],
        default="xesmf",
        help="Regridding backend to use.",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=96,
        help="Number of levels (if generating data); default 96",
    )
    p.add_argument(
        "--lat",
        type=int,
        default=720,
        help="Source grid latitude points (if generating data); default 720",
    )
    p.add_argument(
        "--lon",
        type=int,
        default=1440,
        help="Source grid longitude points (if generating data); default 1440",
    )
    p.add_argument(
        "--lat-new",
        type=int,
        default=360,
        help="Target grid latitude points; default 360",
    )
    p.add_argument(
        "--lon-new",
        type=int,
        default=720,
        help="Target grid longitude points; default 720",
    )
    p.add_argument(
        "--chunk-level",
        type=int,
        default=1,
        help="Chunk size along level (if generating data); default 1",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of workers for threads/processes; default 24",
    )
    p.add_argument(
        "--scheduler",
        choices=["sync", "threads", "processes", "distributed", "all"],
        default="all",
        help="Which scheduler(s) to run: one of sync, threads, processes, distributed, or all (default)",
    )
    p.add_argument(
        "--demo-threads",
        action="store_true",
        help="Use preset where threads usually win: levels=64, 1440x720->720x360, chunk_level=1",
    )
    args = p.parse_args()

    if args.demo_threads:
        args.levels = 64
        args.lat = 1440
        args.lon = 720
        args.lat_new = 720
        args.lon_new = 360
        args.chunk_level = 1
        print(
            "Using --demo-threads preset (threads-friendly: many tasks, moderate 2D, no process copy).\n"
        )

    if args.scheduler == "all":
        schedulers = ["sync", "threads", "processes", "distributed"]
    else:
        schedulers = [args.scheduler]

    run_benchmark(
        input_zarr=args.input_zarr,
        n_level=args.levels,
        n_lat=args.lat,
        n_lon=args.lon,
        n_lat_new=args.lat_new,
        n_lon_new=args.lon_new,
        chunk_level=args.chunk_level,
        n_workers=args.workers,
        schedulers=schedulers,
        regridder_name=args.regridder,
    )


if __name__ == "__main__":
    main()

