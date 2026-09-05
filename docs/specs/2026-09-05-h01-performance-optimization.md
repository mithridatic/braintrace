# H01 Performance Optimization and Resource Reduction

Date: 2026-09-05
Status: In Progress

## 1. Problem Statement & Motivation
Simulations of human cortical reconstructions (H01) with active multi-compartment Hodgkin-Huxley channels (e.g. Wilbers 2023, HL5BN1 PV reference, Allen L2/L3 pyramidal) exhibit high compilation latency and CPU overhead:
- Hodgkin-Huxley channel mechanisms query individual gate properties (`f_m_inf`, `f_m_tau`, `f_h_inf`, `f_h_tau`) independently during solver advancement.
- In existing implementations, each gate accessor invokes `self._rates(...)` which computes the entire dictionary of all gates in that mechanism (often computing 4+ `rate_trap` calls per evaluation). For a 2-gate channel, this leads to 4x redundant calculations per compartment per time step.
- `rate_trap` contains nested conditional branches (`jnp.where`) and series approximations that inflate JAX trace graph sizes (~8,885 primitive eqns per step for 327 compartments).
- Memory allocations during long simulation rollouts can be streamlined.

## 2. Technical Objectives & Constraints
1. **Mathematical Equivalence:** Rate equations, time constants, equilibria, steady-state limits, and derivatives must remain identical to published source kinetics.
2. **Backward Compatibility:** All existing function signatures (`pv_rates`, `l2_rates`, `sodium_rates`, `potassium_rates`) must remain backwards compatible and return full dictionaries when called without gate filters.
3. **Graph & Runtime Efficiency:**
   - Eliminate redundant gate computations in `_rate_accessor` by querying only the relevant gate rates.
   - Streamline `rate_trap` expressions.
   - Reduce JAX equation count per step.
4. **Test Coverage:** All 222+ unit and integration tests must pass element-wise with exact numerical precision.
