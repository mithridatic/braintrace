# H01 Solver Memory Elimination and Fast Network Execution

Date: 2026-09-06. Branch: feat/h01-braincell.

## Context and Problem Statement

The verified four-neuron, two-synapse H01 network (75,605 total compartments)
constructs successfully, but simulation execution was blocked during solver
initialization (`Network.init_state`).

1. **Solver Preparation MemoryError**: `Cell.init_state` invokes
   `build_cv_axial_operator`, which allocates a dense $(N_{\text{point}} \times N_{\text{point}})$
   axial matrix and attempts dense algebraic node elimination via
   `np.linalg.solve(algebraic_algebraic, algebraic_dynamic)`. For cell 3955003482
   ($N_{\text{cv}} = 33,965$, $N_{\text{point}} \approx 64,047$, $N_{\text{alg}} \approx 30,082$),
   this dense solve attempts to allocate $>30$ GB of memory and fails with `MemoryError`.
   Furthermore, this dense $N_{\text{cv}} \times N_{\text{cv}}$ operator is never
   used during staggered DHS time-stepping (`dhs_voltage_step`), which operates
   directly on the tree representation.

2. **DHS Static Source Memory Allocation**: `_build_dhs_static_source` in the
   installed BrainCell package constructs a full dense $(N_{\text{point}} \times N_{\text{point}})$
   matrix solely to extract diagonal and parent-child off-diagonal elements.
   On large trees, this allocates tens of gigabytes unnecessarily.

3. **JAX Compilation Bottleneck**: The unrolled loop in `comp_triang_raw`
   unrolls thousands of tree levels in Python, resulting in massive XLA graph sizes
   and lengthy compilation times. Scanned tree elimination lowers this into a
   single compiled scan primitive.

4. **Initialization Redundancy**: `Cell.init_state` invalidates the discretization
   cache during morphology cloning, re-discretizing each cell from scratch, and
   performs millions of scalar unit checks during runtime geometry attachment.

## Proposed Architecture and Optimization

1. **$O(N)$ Sparse DHS Static Source Assembly**:
   Directly assemble 1D `diag_ms_inv`, `lowers_ms_inv`, and `uppers_ms_inv` arrays
   from `node_tree.edges` in $O(N)$ time and memory. The mathematical values
   match the dense extraction identically.

2. **Deferred / Lazy Axial Operator**:
   In `H01Cell`, defer dense `build_cv_axial_operator` computation during `init_state`.
   Staggered DHS time-stepping uses the 1D tree source directly. If `compute_axial_derivative`
   or `axial_operator_np` is explicitly queried, compute it on demand.

3. **Compiled Scanned DHS Solver Integration**:
   Integrate the scanned DHS tree elimination (`h01_dhs_scan`) into the default
   execution pipeline for H01 cells, ensuring fast XLA compilation and execution
   without unrolled loop overhead.

4. **Optimized Discretization and Runtime State Initialization**:
   In `H01Cell.init_state`, reuse precomputed discretization structures and vectorize
   geometry scatter and unit conversions.

## Verification Plan

1. **Exact Parity Checks**:
   Verify that 1D sparse DHS source assembly produces bitwise/numerical identical
   `diag`, `lowers`, and `uppers` against dense extraction on test fixtures.
   Verify that voltage time courses match the standard solver to within machine tolerance.

2. **Large Circuit Execution**:
   Run `examples/h01_verified_network.py` with the full 4-neuron circuit for finite
   durations (e.g. 0.005 ms, 0.05 ms, and multi-step runs) with recorded voltage traces,
   verifying finite voltages and proper synaptic conductance propagation without memory errors.

3. **Unit Tests and Coverage**:
   Co-locate new tests in `h01_construction_test.py`, `h01_network_test.py`, and
   `h01_dhs_scan_test.py`. Maintain >90% line coverage and verify zero regressions
   across the test suite.
