# H01 Runtime Throughput, Memory and Compilation Optimization

Date: 2026-09-09
Author: AdaL <adal@sylph.ai>
Status: In Progress

## 1. Problem Statement & Motivation
Simulation execution and test suite runtimes across H01 connectome circuits (multi-compartment morphologies, Dale-partitioned E/I circuits, and synaptic delivery networks) exhibited heavy latency and memory usage:
1. **Eager JAX Micro-Compilations During Initialization**:
   - Evaluating rate laws (`rate_trap`, `pv_rates`, `l2_rates`, `sodium_rates`, `potassium_rates`) outside JIT during `Cell.init_state`, `reset_state`, and probe setup converted arrays to `jax.Array` and executed dozens of eager JAX operations (`jnp.where`, `jnp.exp`, `jnp.maximum`, `sigmoid`), triggering hundreds of individual XLA micro-compilations (`pjit`) per cell.
2. **Channel-by-Channel Autodiff in Exponential Euler**:
   - `braincell.quad.ind_exp_euler_step` wrapped every gating variable in `brainstate.transform.vector_grad` to compute local Jacobians via automatic differentiation. For multi-compartment cells with 30+ gating states, this generated thousands of intermediate tracer nodes and nested autodiff equations inside `_step`, causing multi-second XLA compilation overhead for each simulation rollout.
   - For all Hodgkin-Huxley (`HH`) channels, the derivative w.r.t the gate variable is analytically linear: $\frac{d}{dg}(\frac{\phi (\text{inf} - g)}{\tau}) = -\frac{\phi}{\tau}$. The exact exponential Euler step $g_{t+1} = g_t + \Delta t \cdot \varphi_1(-\Delta t \frac{\phi}{\tau}) \cdot \dot{g}_t$ can be evaluated directly without autodiff overhead.
3. **Membrane Voltage Linearization Redundancy**:
   - `_staggered._linear_and_const_term` dynamically instantiated `vector_grad(target.compute_membrane_derivative)` on each step when `_voltage_linearizer` was not cached on the target cell.
4. **Repeated Gating Equilibrium Calculations in Phase Scaling**:
   - In `_rate_accessor`, both `inf` and `tau` recomputed equilibrium rates repeatedly without sharing sub-expressions.

## 2. Technical Objectives & Constraints
1. **Bitwise and Numerical Parity**:
   - All membrane voltages, gating states, synaptic conductances, calcium dynamics, and spike events must match existing baselines element-wise within floating-point tolerance ($< 10^{-12}$).
2. **Reduced Peak Memory & Zero Micro-Compilations**:
   - Eliminate eager JAX compilations during non-traced cell and network initialization by using pure NumPy / C arithmetic when inputs are not JAX tracers.
3. **Drastic Speedup for Simulations and Tests**:
   - Accelerate `Network.run`, `Cell.run`, and all 484+ H01 tests by bypassing redundant `vector_grad` graph construction for analytical `HH` channels and `DynamicNernstIon` instances.
4. **Full Test Suite Validation**:
   - All unit and integration tests across `braintrace/datasets/` and `examples/` must pass cleanly.

## 3. Architecture & Implementation Plan
1. **Dual-Path Rate Laws**:
   - Update `h01_wilbers.py`, `h01_pv_rates.py`, and `h01_l2_channels.py` to check `is_traced_value`. When inputs are floats or NumPy arrays (un-traced), execute fast NumPy operations (`np.exp`, `np.where`, `np.expm1`, `expit`). When inputs are JAX tracers, execute JAX expressions (`jnp.exp`, `jnp.where`, `jnp.expm1`, `sigmoid`).
2. **Fast Analytical Exponential Euler Integration for HH Channels**:
   - Enhance `_exp_euler._ind_exp_euler_step_selected` (and hook it into `h01_dhs_scan.py` and `H01Cell`) to recognize `HH` channels and evaluate the exact analytical exponential Euler update directly, bypassing `vector_grad`.
3. **Optimized Voltage Linearizer & State Initialization**:
   - In `H01Cell.init_state`, initialize gating variables with fast NumPy mantissas and provide a pre-bound `_voltage_linearizer` on `H01Cell` instances.
4. **Verification & Profiling**:
   - Profile before and after with `cProfile` and run the entire test suite.
