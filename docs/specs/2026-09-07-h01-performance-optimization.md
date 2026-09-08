# H01 Runtime Performance and Resource Optimization

Date: 2026-09-07
Author: AdaL <adal@sylph.ai>
Status: Completed

## 1. Problem Statement & Motivation
Simulation initialization, geometry projection, and DHS time-stepping across H01 connectome circuits (multi-compartment morphologies with up to tens of thousands of compartments per neuron) exhibited major execution bottlenecks:
1. **DHS Scan Kernel Scans & Transformations**: `_triang` and `_backsub` used `brainstate.transform.scan` for internal tree updates, incurring state-cloning graph walks and heavy XLA HLO module generation within the simulation loop.
2. **Dynamic Backend Import Overhead**: `saiunit._backend` checked `is_torch_array` via `_try_import("torch")` on every scalar check during parameter initialization, causing seconds of unnecessary DLL loading latency on Windows.
3. **Ion Runtime Instantiation & Geometry Attachment Overhead**: `_build_runtime_ions` constructed full-sized baseline ions solely to inspect scalar default parameters, and `attach_runtime_ion_geometry` looped over all CV objects with individual scalar Quantity conversions.
4. **Redundant Discretization Invalidation**: `H01Cell.init_state` previously called `_invalidate_discretization_cache()`, forcing complete re-discretization of multi-thousand compartment morphology trees even when morphology geometry was unchanged.
5. **Dense Axial Matrix Allocations in Donor/Reference Cells**: `make_l2_cell`, `make_pv_cell`, and `make_reference_cell` instantiated base `braincell.Cell` directly, computing $O(N^2)$ dense axial matrices rather than using the 1D sparse DHS tree source in `H01Cell`.
6. **Anatomical Graph & Geodesic Recomputation**: `H01Anatomy.cable_neighborhood` rebuilt the source adjacency graph and recomputed segment lengths on every call. `H01Anatomy.project` performed full-array square roots across all cable segments for every query point.

## 2. Technical Objectives & Constraints
1. **Exact Parity**: All numerical results, voltages, conductances, spike events, and geometric interval definitions remain identical to existing baselines within floating-point tolerance ($< 10^{-12}$).
2. **Resource Efficiency**: Drastically reduce memory allocations and initialization time during cell/network setup.
3. **Speed**: Accelerated DHS solver scanning, cell initialization, backend dispatch, and anatomical queries.
4. **Full Test Suite Validation**: All 341 tests across unit and integration suites pass element-wise with exact precision.

## 3. Implementation Details
1. **Pure JAX Scanned DHS Kernels (`_triang` and `_backsub`)**:
   - Replaced `brainstate.transform.scan` with `jax.lax.scan` directly operating on raw array mantissas and reconstituting units on exit.
   - Reduced DHS scan JIT compilation and execution latency from 9.27s down to 0.265s (>35x speedup).
2. **Dynamic Backend Import Guard**:
   - Guarded `saiunit._backend._try_import` to prevent loading unimported heavy backends (`torch`, `cupy`, `dask`, `ndonnx`) during scalar unit checks.
3. **Fast Ion Baseline Inspection & Vectorized Geometry Attachment**:
   - Properly patched `_runtime_module.attach_runtime_ion_geometry` and `_runtime_module._instantiate_runtime_ion_instance`.
   - Extracted mantissa arrays in bulk without per-CV scalar Quantity conversions.
4. **Cached Morphology Discretization in `H01Cell.init_state`**:
   - Preserved discretization cache and `_h01_geom_sig` during `H01Cell.init_state`, eliminating redundant tree discretization.
5. **Sparse DHS Tree Assembly for L2/PV Reference Cells**:
   - Updated `make_l2_cell`, `make_pv_cell`, and `make_reference_cell` to use `H01Cell` with 1D sparse DHS static source assembly, avoiding dense axial matrix construction.
6. **Precomputed Anatomy Adjacency & Optimized Projections**:
   - Precomputed `_lengths`, `_row_id_to_index`, and `_adjacency` during `H01Anatomy.__init__`.
   - Optimized `H01Anatomy.project` by filtering candidate segments using squared distance tolerance ($2 \cdot d \cdot \epsilon + \epsilon^2$), avoiding full-array `np.sqrt` calls.

## 4. Verification & Results
- Profiled with `cProfile` and `line_profiler`.
- Verified exact numerical and geometric parity across all tests.
- All 341 tests in `braintrace/datasets/*h01*test*.py` and `examples/h01*.py` pass.
