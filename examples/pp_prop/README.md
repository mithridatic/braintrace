# pp_prop Examples

A tutorial-linear walk through `braintrace.pp_prop` (aliases `ES_D_RTRL` /
`IODimVjpAlgorithm`) — an online eligibility-trace gradient estimator with
input-output dimensional complexity for spiking neural networks. Examples 01–16
form the self-contained tutorial sequence; later investigations may use sibling
support modules and co-located tests. Read 01 → 16 in order to follow the
companion tutorial at `docs/tutorials/pp_prop.ipynb`.

## How to run

    python examples/pp_prop/01-basics-lif-integrator.py

Most introductory fixed-size examples run on CPU. Device-aware Examples 16, 17,
20, and 21 document their defaults below. Example 20 defaults to GPU and fails
closed when JAX cannot bind one; pass `--device cpu` for an explicit host run.
The digit examples require the examples extra:

    pip install "braintrace[examples]"

Example 16 is a configurable scaling benchmark, so its runtime depends on the
requested neuron count and update budget.

## Axis map

| Axis                                      | Files              |
|-------------------------------------------|--------------------|
| Neuron model (LIF / ALIF / GIF / COBA-EI) | 01, 02, 03, 04     |
| Batching mode (vmap vs batched primitive) | 05, 06             |
| vjp_method (single-step vs multi-step)    | 07, 08, 14         |
| Operator (matmul / sparse / LoRA / conv)  | 09, 10, 11         |
| Training target                           | 01, 02, 03, 04, 12 |
| Algo knob (decay vs rank)                 | 13                 |
| BPTT baseline                             | 12, 14             |
| Held-out learning evidence                | 15                 |
| Configurable sparse scaling               | 16                 |
| Delayed-cue temporal credit               | 17                 |
| Topology (fixed vs evolved)               | 18                 |
| Post-training topology analysis           | 19, 20             |
| Standard ARC exact-grid inference         | 21                 |
| Same-model variable latent effort         | 21                 |
| Latent trajectory and causal controls     | 21                 |

### File-by-file summary

| #  | File                                | Demo                                                    |
|----|-------------------------------------|---------------------------------------------------------|
| 01 | `01-basics-lif-integrator.py`       | LIF RSNN on Poisson-to-cumulative-rate regression       |
| 02 | `02-neurons-alif-dms.py`            | ALIF (adaptive threshold) on delayed-match-to-sample    |
| 03 | `03-neurons-gif-working-memory.py`  | GIF with heterogeneous tau_I2 on working-memory recall  |
| 04 | `04-neurons-coba-ei-rsnn.py`        | Dale-law E/I RSNN on small Poisson-MNIST                |
| 05 | `05-batching-vmap.py`               | Batching via `brainstate.nn.Vmap(vmap_states='new')`    |
| 06 | `06-batching-batched.py`            | Batching via the batched ETP primitive path             |
| 07 | `07-vjp-single-step.py`             | `vjp_method='single-step'` (default)                    |
| 08 | `08-vjp-multi-step.py`              | `vjp_method='multi-step'` for temporal credit           |
| 09 | `09-operator-sparse.py`             | Native CSR recurrent connectivity with SparseLinear     |
| 10 | `10-operator-lora.py`               | Low-rank recurrence via `braintrace.lora_matmul`        |
| 11 | `11-operator-conv.py`               | Conv-SNN via `braintrace.nn.Conv2d`                     |
| 12 | `12-classification-neuromorphic.py` | Small pp_prop and BPTT classifier smoke comparison      |
| 13 | `13-knob-decay-vs-rank.py`          | Sweep `decay_or_rank` across floats and ints            |
| 14 | `14-knob-vjp-method-contrast.py`    | single-step vs multi-step vs BPTT head-to-head on DMS   |
| 15 | `15-sparse-temporal-learning.py`    | Sparse LIF learning on held-out handwritten digits      |
| 16 | `16-configurable-sparse-benchmark.py` | Guarded synthetic sparse-CSR scaling and target timing |
| 17 | `17-temporal-credit-benchmark.py` | Paired delayed-cue recall and recurrent-credit evidence |
| 18 | `18-structural-evolution.py`      | Two-trick continual learning with prune/regrow evolution |
| 19 | `19-structural-evolution-cfsg-symmetry.py` | Topology-only twin symmetry and task-attribution analysis of Example 18 |
| 20 | `20-post-training-neuron-pruning.py` | Joint causal neuron/edge lesions and a coordinate-wise locally minimal network after Example 18 |
| 21 | `21-braincell-arc.py` | Standard ARC grids through the BrainCell ARC replacement |

### Post-training neuron-and-edge pruning

Example 20 defaults to Example 18's four-task temporal-credit configuration and
starts pruning at the first pre-rebuild checkpoint where every task meets the
requested target. After a coarse-to-fine starting sweep, it individually tests
retained neurons and retained-to-retained recurrent edges, accepts safe
removals, reranks, and alternates both phases until neither can remove another
coordinate. It then physically rebuilds the surviving feed-forward, recurrent,
and readout arrays into a compact inference model, verifies its logits against
the masked checkpoint, and saves a reloadable NumPy bundle. A larger sparse
starting graph can be requested directly:

    python examples/pp_prop/20-post-training-neuron-pruning.py --neurons 2048 --initial-edges 16384 --n-rounds 12 --compact-model-output compacted_network.npz

The default device is GPU. `--device gpu` refuses silent CPU fallback,
`--device cpu` deliberately pins the host, and `--device auto` accepts whatever
JAX selects. For repeated container runs, mount a persistent directory and set
`JAX_COMPILATION_CACHE_DIR` to that mount; setting
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` also retains short compilations.
The fixed-point phase can be much slower than the initial sweep because every
retained neuron and edge receives a causal ablation test in the terminal
passes. The report separates warmed full-probe timing from compilation and
reports persistent parameter-plus-CSR storage for the masked and compact
models.

### Example 21 BrainCell ARC

Example 21 uses the direct BrainCell ARC path. It reads only the named raw ARC
practice files, uses compiled BrainTrace PP-Prop event execution, and returns
direct integer-grid predictions with strict exact results. No generated index or
source manifest is needed.

Run the focused CPU smoke check with:

    python examples/pp_prop/21-braincell-arc.py --smoke --device cpu --output-dir var/example21-smoke

The Example 21 image copies raw ARC files to `/datasets/arc/raw`, installs
`braincell==0.1.0`, and uses `/opt/braintrace` as its source root. Its default
command is:

    python /opt/braintrace/examples/pp_prop/21-braincell-arc.py --help

Run the bounded proof command in the image with:

    python /opt/braintrace/examples/pp_prop/21-braincell-arc.py proof --device gpu

The proof is mechanism evidence, not a claim of ARC skill. Routine commands
use the fixed practice-task order. Evaluation data is outside ordinary
training, structure, and timing runs.

#### Example 21 image run

Build the image with raw ARC data mounted as the `arc_data` stage, then run the
same BrainCell entry point:

    docker run --rm --gpus all \
      braintrace-example21:latest \
      python /opt/braintrace/examples/pp_prop/21-braincell-arc.py --help

### Configurable benchmark

Run the default guarded GPU target search at 32,768 neurons and 100 percent:

    python examples/pp_prop/16-configurable-sparse-benchmark.py

Run an explicit fixed-work measurement:

    python examples/pp_prop/16-configurable-sparse-benchmark.py --mode fixed-work --neurons 131072 --degree 8 --updates 3

Measure the first validation checkpoint at or above 95 percent:

    python examples/pp_prop/16-configurable-sparse-benchmark.py --mode validation-target --neurons 4096 --target-accuracy 0.95 --json-output pp-prop-4096.json

Use ``--help`` to configure temporal steps, final supervision window, batch
size, optimizer settings, trace decay, evaluation cadence, sparse backend,
recurrent scaling basis, and resource limits. Each run uses a fresh worker
process and prints one schema-versioned JSON result. The default wall-clock
limit is 30 minutes. Progress goes to stderr.

New runs emit schema 2 with
`topology_family="legacy_translated_offsets"`, the realized self-loop count,
and an explicit note that those loops are seed-dependent. Existing schema-1
artifacts remain valid and the supervisor continues to parse them.

Run on an accelerator, refusing to fall back to the host:

    python examples/pp_prop/16-configurable-sparse-benchmark.py --device gpu --neurons 131072 --degree 8 --updates 3

`--device` takes `auto` (whatever JAX binds), `cpu` (pins the host backend, so a
GPU host can still measure the CPU arm) and `gpu` (the default, requiring an
accelerator). A default run on a host with no accelerator plugin exits nonzero
with `requested device gpu, bound backend is cpu` rather than reporting host
timings under an accelerator heading. Installing the CUDA plugin is what makes
an accelerator available; `--device gpu` only refuses to proceed without one.

This is a synthetic fixed-degree CSR classifier benchmark with trainable dense
input and readout projections. It is not a connectome-learning benchmark.
Time-to-target repeatedly checks the validation split and is therefore an
adaptive validation metric, not an unbiased held-out estimate. Reported memory
covers both sides of the device boundary: `peak_rss_bytes` is the highest 100 ms
sampled host process-tree RSS, and `device_peak_bytes` is the XLA allocator peak
live allocation, which is null on backends that report no statistics, the host
backend among them. The two are not comparable and neither is a total.

Cross-reference: for the `fast_solve` knob (shared with D_RTRL but not
required for pp_prop), see `examples/drtrl/11-knob-fast-solve.py`.

### Temporal-credit benchmark

Example 17 uses a committed 12-bundle seed manifest and seven matched arms. Its
response window contains only a label-independent go cue. Development runs must
pass `--allow-dirty`; sealed-test runs fail closed on a dirty source tree.

    python examples/pp_prop/17-temporal-credit-benchmark.py --device cpu --horizon short --updates 2 --neurons 24 --degree 4 --allow-dirty

Run the gated short-to-medium-to-long curriculum and the fixed 24-neuron
gradient reference on an accelerator:

    python examples/pp_prop/17-temporal-credit-benchmark.py --device gpu --arm all_pp_prop --curriculum --gradient-evidence --allow-dirty --json-output temporal-credit.json

See `docs/specs/2026-08-10-temporal-credit-benchmark.md` for the claim boundary,
statistical gates, sealed analysis, and release policy. Development output is
not scientific evidence; omit `--allow-dirty` and pass `--sealed-test` only on
the accepted clean commit.

## Tutorial

See `docs/tutorials/pp_prop.ipynb` for the long-form narrative.

## Tests

    pytest examples/pp_prop -v
