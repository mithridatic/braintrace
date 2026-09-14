# Real H01 execution investigation

The complete default 104-cell evolution under 30 seconds remains unverified.
This work continues from published `a6b2008`, in the isolated
`perf/h01-event-profile-20260914` branch and `/workspace/h01-event-profile-20260914`
Vast.ai checkout. The single-kernel integration and subsequent qualification
are recorded in `h01-gpu-solver-20260914.md`; regrouping remains experimental.

## Authoritative baseline

The other session's PID 805584 remained live around 58 minutes with no files
under `/tmp/evolve_h01_test`. Its checkout remains pinned to `e7b8f3a`. No signal,
restart, or source change was applied to it. A py-spy stack dump was denied by
the container's missing SYS_PTRACE capability; do not retry by restarting it.

The isolated four-anatomical-cell profile used 75,605 compartments, the current
qualified dt from numerical_settings, 64-bit precision, and 160 substeps per
event. Its completed cProfile phases were construction 32.035 s, initialization
26.656 s, model setup 2.630 s, cold event 27.808 s, warm event 17.936 s, and
sparse graph construction 3.171 s. The warm-event profile spent 17.917 s inside
compiled execution and did not show backend compilation. The first sparse
Muon update did not finish during the five-minute-capped experiment. Its
process was verified missing afterward; raw report status remains `running`
because the external termination did not execute the script's final save.
Do not interpret that report as a learning pass. The SSH handle returned 1,
so an exact timeout exit-code claim is not supported.

## Experiments

1. Invalid scatter entries now address one past the buffer and explicitly
   drop updates, rather than atomically adding zero to one valid sentinel.
   This candidate passed 22 xdist solver tests, including dense oracles,
   implicit forward/reverse derivatives, real E/I traces, and circuit events.
   Uneven synthetic-tree elimination improved from 3.77–3.94 ms to
   2.35–2.57 ms. The real four-cell event changed only from 17.936 to 17.665 s;
   shared-GPU noise prevents a strong real-event speed claim.
2. Actual scheduling arrays have shapes (3232,32), (1496,32), (1914,32), and
   (2505,32). BrainCell's default max_group_size=32 creates these batches.
   True dependency heights are 1568,1043,1341,2091. A single rectangular
   dependency-height pack was rejected: it expands padding dramatically and
   slows three of four trees.
3. Width-bounded contiguous dependency-height groups reduce padding to
   72395,24228,29057,25149 entries, versus 103424,47872,61248,80160. Pure linear
   solve timings improved from 37.64/16.60/22.92/30.33 ms to
   17.01/11.96/13.14/21.17 ms with zero observed output error. This remains an
   evidence-only prototype; its gradients and full physical path are unqualified.
4. A one-warp Pallas/Triton kernel executes the unchanged 32-wide elimination
   schedule inside one GPU kernel. Warm elimination timings were
   39.19/16.77/21.06/31.37 ms before and 6.39/2.78/5.39/5.44 ms after, with
   zero observed errors on those four synthetic-coefficient anatomical-tree
   systems. This is the leading next candidate. It is not integrated into the
   physical solver or training and has no gradient qualification yet.

The Pallas prototype uses the installed JAX 0.11 reference-view load API
`plt.load(ref.at[index])`, atomic adds with a validity mask, one warp, and a
barrier between levels. An initial old-style load signature failed before
execution and was corrected after inspecting installed source. Do not silently
replace the barrier, precision, or implicit derivative rule.

## Next execution gate

Integrate the single-kernel elimination only inside custom_linear_solve's
primal/transpose solver callbacks, with a portable existing-kernel fallback.
Qualify float64, batching/vmap, sentinel preservation, input non-mutation,
forward/reverse derivatives against dense solves, and real H01 finite-window
learning. Measure the complete real event again before default activation.
Current 22-test results qualify only the invalid-scatter candidate.

Reproducible scripts are h01_event_profile.py, h01_padding_profile.py,
h01_tree_group_profile.py (the width-bounded variant), and h01_pallas_profile.py.
Actual tree exports remain at `/tmp/h01-real-trees-20260914` on Vast.ai for cheap
subsequent kernel experiments. No model is repeatedly driven through an
uncompiled Python loop. The pure numerical kernel probes use JAX jit.

These are small samples on a GPU shared with the other session. No full
evolution runtime, test-suite-wide improvement, or ARC quality claim follows.
