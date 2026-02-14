# Radio-Astro Regridding Correctness Plan

## Scope
This plan targets 2D spatial regridding of radio astronomy images/cubes where the regridded axes represent sky coordinates (for example `l,m` or `RA,Dec`) and pixel values represent flux density (for example `Jy/beam` or `Jy/pixel`).

Primary objective: scientific correctness over runtime performance.

## 1. Define Quantity Semantics Before Regridding
For each product, classify the pixel quantity and apply the matching rule:

- `Jy/pixel` (or equivalent integrated-per-pixel quantity): total flux over footprint should be conserved (within tolerance).
- `Jy/beam`: do not enforce naive pixel-sum conservation unless beam treatment is handled; validate peak and morphology with beam-aware checks.
- Dimensionless or mask-like products: validate categorical/mask integrity separately.

Required metadata to record per dataset:
- Unit string.
- Beam model (major/minor/PA) or statement that no beam model applies.
- WCS frame/projection metadata.
- No-data/mask convention.

## 2. Geometry/WCS Correctness Checks
Do not treat sky coordinates as generic Cartesian labels without verification.

Pre-checks:
- Verify monotonic coordinate axes and valid WCS transform.
- Validate RA wrap behavior near 0/360 (or equivalent discontinuity handling).
- Confirm output footprint overlap with input footprint.

Pass criteria:
- No coordinate discontinuity artifacts at wrap boundaries.
- Forward/inverse transform round-trip error below `1e-6` pixel at sampled points (or stricter project-specific value).

## 3. Algorithm Selection Rules (Science-Driven)
Choose algorithm by measurement objective:

- Flux/integrated quantity studies: use conservative or area-aware remapping.
- Morphology-focused visualization where conservation is less critical: bilinear/cubic may be acceptable with documented tradeoff.
- Never assume `xarray.interp` and `xESMF` outputs are equivalent on arbitrary grids.

Required run log fields:
- Backend and method.
- Extrapolation/fill policy.
- Boundary and periodic settings.

## 4. Beam/PSF Handling
For `Jy/beam` workflows:
- Harmonize to common beam before cross-grid comparisons when scientifically required.
- Track beam metadata through regridding and update downstream interpretation accordingly.

Pass criteria:
- Beam metadata present and consistent with operation history.
- Peak/intensity comparisons only made after beam-consistent preprocessing.

## 5. Masks, NaNs, and Footprint Policy
Define explicit rules and test them:

- Preserve invalid/flagged regions.
- Do not replace NaNs with zeros unless explicitly required and documented.
- Set and test out-of-footprint fill behavior.

Pass criteria:
- Mask leakage rate <= `0.1%` of output pixels (or tighter project threshold).
- No unexpected valid values outside overlap footprint.

## 6. Reference Test Suite (Must-Have Cases)
Run on small deterministic fixtures plus at least one realistic cube subset.

Synthetic fixtures:
- Single point source at known coordinates.
- Elliptical Gaussian source with known integrated flux.
- Uniform field (constant surface brightness).
- Two-source blend with known centroid separation.
- Case crossing coordinate wrap boundary.

Real-data fixture:
- Representative subcube spanning multiple channels.

## 7. Quantitative Acceptance Metrics
Evaluate both backend-vs-reference and regression-vs-baseline.

Core metrics:
- Integrated flux error over valid footprint.
- Peak flux error.
- Centroid offset (pixels and angular units).
- RMS residual over overlap area.
- Mask disagreement fraction.

Suggested default thresholds (tune per science case):
- Integrated flux error: `<= 1%` for conserved-quantity workflows.
- Peak flux error: `<= 2%` for beam-consistent comparisons.
- Centroid shift: `<= 0.1` pixel.
- Mask disagreement: `<= 0.1%`.

## 8. Cube Consistency Checks
Validate across non-spatial axes (frequency/Stokes/time):

- Run metrics on multiple channels, including band edges.
- Ensure no channel-dependent bias trend introduced by regridding.
- Validate per-Stokes behavior if polarization products are present.

Pass criteria:
- No monotonic drift in error metrics across channel index beyond expected noise/systematics.

## 9. Cross-Tool Validation
For release gating, compare against at least one trusted external astro toolchain (for example `astropy/reproject` or CASA-equivalent workflow) on the same fixtures.

Pass criteria:
- Agreement within the project thresholds above, or documented and justified deviations tied to method differences.

## 10. Reproducibility and Provenance
Every validation run should persist:
- Input dataset IDs/checksums.
- Code commit SHA.
- Full regridding configuration (backend/method/params).
- Environment versions (`xarray`, `dask`, `xesmf`, `numpy`, `scipy`, astro tooling).
- Metric results and pass/fail verdict.

## 11. Minimal Implementation Checklist for This Repository
- Generalize spatial axis handling so backend path does not assume `lat/lon` naming.
- Add automated tests for synthetic fixtures and metric thresholds.
- Add a command/script to run the full correctness suite and emit machine-readable results.
- Gate backend changes on correctness suite pass, not benchmark speed alone.

