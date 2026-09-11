# Performance and Memory Optimization for BrainTrace H01 Pipeline, Sparse PP-Prop, and Example 21

Date: 2026-09-11
Author: AdaL <adal@sylph.ai>
Status: Approved

## 1. Executive Summary & Objectives
The H01 multicompartment connectome pipeline, Sparse PP-Prop online learning algorithm, and Example 21 BrainCell ARC training pipeline exhibited severe resource contention, high wall-clock latency, and memory bloat during execution and test runs:
- `h01_arc_probe` took >400s wall-clock on GPU (Dutch Vast.ai RTX 4090 instance), with major bottlenecks in cell construction (26-44s), cell state initialization (39-53s), and Sparse PP-Prop/Muon JAX compilation and execution (87-130s).
- Repeated redundant morphology and geometry validations over 30,000+ branches per cell during discretization.
- Eager JAX GPU dispatches during cell initialization creating dozens of device synchronization round-trips per cell.
- In Sparse PP-Prop (`advance_factors`, `contract_factors`), redundant dynamic tensor allocations (`jnp.zeros().at[...].set(...)`) and re-evaluating static color indexing during inner scan steps slowed compilation and execution.
- Test suite running multi-threaded OpenBLAS workers on 64-thread host caused thread exhaustion and memory pressure.

This optimization addresses these root causes to drastically speed up execution, reduce host RAM and GPU memory usage, and achieve fast test execution while guaranteeing 100% numerical and bitwise parity.

## 2. Technical Scope & Architecture Changes

### A. H01 Construction & Geometry Discretization (`braintrace/datasets/h01_construction.py`, `braintrace/datasets/h01.py`)
1. **Validation Bypassing**:
   - Skip redundant morphology and bounds re-validation in `_fast_build_cv_geometry` when `morpho._h01_validated` is True.
2. **Optimized Multi-Segment Frusta & Static Geometry Evaluation**:
   - For multi-segment branches with full `(0.0, 1.0)` coverage, compute lateral surface area and axial factor total via vectorized NumPy operations rather than iterative single-frustum constructions and splitting.
3. **Discretization Caching Across Equivalent Declarations**:
   - Cache `Discretization` instances by `(morpho_signature, policy, paint_rules, place_rules)` so that `cell.init_state()` directly retrieves the pre-built CV tree and node tree.
4. **Fast Pure NumPy State Initialization**:
   - In `H01Cell.init_state`, initialize membrane voltage `V`, threshold `V_th`, capacitance `C`, and initial gating states as concrete NumPy arrays before wrapping in `DiffEqState` and `ShortTermState`, completely eliminating eager host-device syncs and JAX GPU kernel dispatches during network initialization.
5. **Fast SWC Normalization and Branch Extraction**:
   - Optimize SWC parsing in `_fast_build_morpho_from_text` by eliminating repetitive string rule scans over already normalized and verified SWC arrays.

### B. Sparse PP-Prop & Matrix-Free Factor Propagation (`braintrace/_algorithm/sparse_io.py`, `braintrace/_algorithm/sparse_pp_prop.py`)
1. **Static Precomputation of Output Seeds & Color Layouts**:
   - Precompute constant `out_seeds` matrix `(1.0 - decay) * (colors_np[None, :] == np.arange(color_count)[:, None])` on the layout structure rather than constructing it dynamically within the scan step.
2. **Streamlined Hidden Seed Scatter in `advance_factors`**:
   - For blocks matching all colors or single colors, avoid allocating dense zero buffers with `.at[...].set(...)`.
3. **Vectorized Contraction in `contract_factors`**:
   - Vectorize factor cotangent reduction across state blocks.

### C. Example 21 & Trainer Enhancements (`examples/pp_prop/21-braincell-arc.py`, `examples/pp_prop/example21_arc_adapter.py`)
1. **Efficient Parameter Syncing & Gradient Grouping**:
   - Pre-cache parameter path mappings in `PPPropEpisodeTrainer._group_gradients` and `_sync_compiled_parameters`.
2. **Optimized Episode Step Execution**:
   - Ensure compiled step and update functions avoid unnecessary tensor copies and host allocations.

## 3. Verification & Acceptance Criteria
1. **Probe Pass**: `python -m examples.h01_arc_probe --cache /h01 --cells 12 --sparse-learning` passes all 9 phases with status `learning_probe_pass`.
2. **Example 21 Smoke & Proof**: `python examples/pp_prop/21-braincell-arc.py --smoke --device gpu` passes cleanly.
3. **Full Test Suite**: All tests pass cleanly without errors.
4. **Significant Speedup**: Substantial reduction in total runtime and RAM usage across all phases.
