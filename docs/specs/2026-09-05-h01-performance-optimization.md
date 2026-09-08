# H01 Performance Optimization and Resource Reduction

Date: 2026-09-05
Status: Completed

## 1. Problem Statement & Motivation
Simulations of human cortical reconstructions (H01) with active multi-compartment Hodgkin-Huxley channels (e.g. Wilbers 2023, HL5BN1 PV reference, Allen L2/L3 pyramidal) exhibited high compilation latency and execution overhead:
- Hodgkin-Huxley channel mechanisms queried individual gate properties (`f_m_inf`, `f_m_tau`, `f_h_inf`, `f_h_tau`) independently during solver advancement.
- Each gate accessor invoked `self._rates(...)` which computed the entire dictionary of all gates in that mechanism (evaluating 4+ `rate_trap` calls and multiple transcendentals per evaluation). For a 2-gate channel, this led to 4x redundant calculations per compartment per time step.
- Furthermore, querying equilibrium steady state (`component=0`, e.g., `f_m_inf`) unnecessarily computed expensive `rate_trap` and Gaussian exponential time constants (`tau`) that were discarded immediately.
- `rate_trap` evaluations contained unnecessary duplicate branches that inflated JAX trace graph sizes (~8,885 primitive equations per step for 327 compartments).

## 2. Technical Objectives & Constraints
1. **Mathematical Equivalence:** Rate equations, time constants, equilibria, steady-state limits, and derivatives remain mathematically identical to published source kinetics.
2. **Backward Compatibility:** All existing function signatures (`pv_rates`, `l2_rates`, `sodium_rates`, `potassium_rates`) remain fully backwards compatible, returning the default full dictionary/tuple when called without gate or component filters.
3. **Graph & Runtime Efficiency:**
   - Eliminated redundant gate computations in `_rate_accessor` by querying only the relevant gate and component (`component=0` for equilibrium, `component=1` for tau).
   - Targeted `sodium_rates` and `potassium_rates` with optional `gate="m"` / `gate="h"` filtering.
   - Streamlined `rate_trap` expressions while preserving exact continuous limits and gradient stability.
   - Reduced JAX equation count per step from ~8,885 down to 5,421 for PV multi-compartment cells (~39% graph reduction) and 4,756 for L2 pyramidal cells.
4. **Test Coverage:** All 235 unit and integration tests pass element-wise with exact numerical precision.

## 3. Profiling Analysis & Results

### 3.1 Profiling Tools Used
- **cProfile**: Captured call hierarchies, cumulative runtime, and Python overhead during compilation and rollout.
- **line_profiler**: Measured line-by-line execution times across rate calculations (`rate_trap`, `sodium_rates`, `potassium_rates`, `pv_rates`, `l2_rates`).

### 3.2 Key Findings
1. **Redundant Rate Evaluations**: In `_rate_accessor`, querying `f_m_inf` and `f_m_tau` independently previously calculated full `_rates()`. For channels with multiple gates (`m`, `h`), every step recalculated all gates 4 times per compartment.
2. **Unneeded Transcendentals in Steady States**: Calculating `inf` only requires algebraic sigmoid evaluation; computing `tau` involves `rate_trap` (with `expm1`) or multi-exponential expressions. Decoupling `component=0` (inf) and `component=1` (tau) eliminated transcendental evaluations during equilibrium queries.
3. **Graph Complexity Reduction**:
   - PV Multi-compartment Cell: 5,421 JAX equations per step.
   - L2 Pyramidal Cell: 4,756 JAX equations per step.

## 4. Verification
- `braintrace/datasets/*h01*test*.py`: 234 unit tests passed.
- `examples/h01_ei_candidates_test.py`: 1 integration test passed.
- Total 235 tests passed in exact numerical precision.
