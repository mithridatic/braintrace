# H01 Runtime Performance and Resource Optimization

Date: 2026-09-07
Author: AdaL <adal@sylph.ai>
Status: Completed

## 1. Problem Statement & Motivation
Simulation initialization, geometry projection, and DHS time-stepping across H01 connectome circuits (multi-compartment morphologies with up to tens of thousands of compartments per neuron) exhibited major execution bottlenecks:
1. **Ion Runtime Instantiation Overhead**: `_build_runtime_ions` constructed a full-sized baseline ion (`size=full_size`) solely to inspect scalar default parameters. For multi-thousand compartment trees, this allocated full DiffEqState structures and parameter buffers multiple times per cell during `init_state`, consuming seconds of initialization latency and tens of megabytes of temporary allocations.
2. **Repeated Geometry Scattering & Unit Conversions**: `attach_runtime_ion_geometry` evaluated 6 geometric attributes (`length`, `area`, `diam_mid`, `diam_arc_mean`, `radius_prox`, `radius_dist`) independently for every ion in the cell, looping over all CV objects and performing scalar Quantity conversions redundantly.
3. **1D DHS Static Source Assembly**: `build_dhs_static_source_1d` processed all tree edges in Python loops with scalar Quantity boxing and `_st._scalar_decimal` unit calls for every parent/child coefficient.
4. **Anatomical Selection & Projection Bottlenecks**: `H01Anatomy.region` used a Python loop across tens of thousands of branch segments to construct interval tuples. `H01Anatomy.project` performed full-array Euclidean norms (`np.linalg.norm`) across all cable segments for every point.
5. **DHS Scan Kernel**: `_staggered_scan_step` used `_cv_to_point` with runtime assertion checks instead of unchecked midpoint mapping.

## 2. Technical Objectives & Constraints
1. **Exact Parity**: All numerical results, voltages, conductances, spike events, and geometric interval definitions remain identical to existing baselines within floating-point tolerance ($< 10^{-12}$).
2. **Resource Efficiency**: Drastically reduce memory allocations and initialization time during cell/network setup.
3. **Speed**: Accelerated `H01Cell.init_state`, `build_dhs_static_source_1d`, `H01Anatomy`, and `_staggered_scan_step`.
4. **Full Test Suite Validation**: 341 tests across unit and integration suites pass element-wise with exact precision.

## 3. Implementation Details
1. **Fast Ion Baseline Inspection (`_fast_instantiate_runtime_ion_instance`)**:
   - Query scalar baseline parameters using cached `size=1` instances instead of `size=full_size`, avoiding massive DiffEqState and BrainState node allocations.
   - Speedup on multi-compartment cell `init_state`: from 11.56s down to 0.53s (>21x faster).
2. **Vectorized Ion Geometry Attachment (`_fast_attach_runtime_ion_geometry`)**:
   - Precompute the 6 point-space geometry quantities once per cell and assign/broadcast to all ions directly in vectorized operations.
3. **Pure-NumPy 1D DHS Static Source Assembly (`build_dhs_static_source_1d`)**:
   - Pre-extract CV axial resistances (in ohm), areas (in cm^2), and capacitances (in uF) as contiguous float64 arrays.
   - Assemble diagonal, lower, and upper coefficients in pure float64 operations without Quantity boxing in the inner edge loop.
4. **Vectorized Anatomy Regions and Projections (`H01Anatomy.region` and `H01Anatomy.project`)**:
   - Vectorized `H01Anatomy.region` selection with boolean mask indexing.
   - Optimized `H01Anatomy.project` with squared distance calculations to avoid full-array square root operations across all segments.
5. **Streamlined Scanned DHS Solver**:
   - Used `_cv_to_point_unchecked` in `_staggered_scan_step`.

## 4. Verification & Results
- Profiled with `cProfile` and `line_profiler`.
- Verified exact numerical and geometric parity across all tests.
- All 341 tests in `braintrace/datasets/*h01*test*.py` and `examples/h01*.py` pass.
