"""19 - CFSG symmetry read on example 18's evolved multi-task wiring.

Example 18 grows/prunes a sparse recurrent net across interleaved tasks and
labels each surviving synapse by which task's gradient dominated it
(task-leaning, or shared). This example runs 18 unmodified, then reads the
resulting topology through the Classification of Finite Simple Groups:

  Two neurons are *twins* if swapping them leaves the (binarized) adjacency
  invariant. Twin classes are orbits of the subgroup of Aut(graph) generated
  by those swaps, a direct product of symmetric groups S_k (one per orbit of
  size k); its Jordan-Holder composition factors are the irreducible
  building blocks of the symmetry the network discovered, and by CFSG each
  factor must be cyclic Z_p, alternating A_n, Lie type, or sporadic.
  Interchangeability symmetry only ever yields Z_2, Z_3, and A_k.

A coarser lens, *role* equivalence by 1-WL color refinement, sandwiches the
same real automorphism group: prod S_k over twin classes <= Aut(graph) <=
prod S_k over refined-role classes.

CFSG as a language for emergent organization:
  - Orbit (twin class)      -> topology-only structural interchangeability
  - Orbit shrinking to 1    -> symmetry breaking: the neuron specialized

18'S own per-edge task attribution gives a second, independent read on the
same question: an orbit is *attribution-pure* if every member's synapses
lean toward the same task (or are all shared); *mixed* if members sit on
different sides of a task boundary despite being structurally
interchangeable. Comparing the evolving arm's orbits against the frozen
control's is the actual experiment: does adaptive grow/prune sharpen
symmetry breaking along task lines beyond what training alone does to a
fixed random topology?

Reuses 18's config, training, and topology machinery entirely (same
pattern 18 uses to import example 09); this file only adds the
post-hoc symmetry analysis. Any of 18's CLI flags apply, e.g. run
``19-structural-evolution-cfsg-symmetry.py --smoke`` for a fast
iteration check.
"""

import importlib.util
import pathlib
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


def _load_structural_evolution():
    path = pathlib.Path(__file__).resolve().with_name("18-structural-evolution.py")
    spec = importlib.util.spec_from_file_location(
        "_pp_prop_structural_evolution", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example 18 from {path}. Check the path and install the required resource.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EX18 = _load_structural_evolution()


# --- Symmetry analysis (twin orbits + WL role refinement) ------------------


def _validate_topology(
    n_rec: int, rows: np.ndarray, cols: np.ndarray
) -> tuple[List[set], List[set]]:
    if n_rec < 1:
        raise ValueError("n_rec must be positive. Set n_rec to a positive value.")
    if rows.ndim != 1 or cols.ndim != 1 or rows.shape != cols.shape:
        raise ValueError("Rows and cols must be aligned one-dimensional arrays. Set Rows and cols to aligned one-dimensional arrays.")
    if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
        cols.dtype, np.integer
    ):
        raise ValueError("Edge endpoints must be integers. Set Edge endpoints to integers.")
    if rows.size and (
        np.any(rows < 0)
        or np.any(rows >= n_rec)
        or np.any(cols < 0)
        or np.any(cols >= n_rec)
    ):
        raise ValueError("Edge endpoint is outside [0, n_rec). Set the named field to a value in the stated range, then rerun the operation.")
    outgoing = [set() for _ in range(n_rec)]
    incoming = [set() for _ in range(n_rec)]
    for row, col in zip(rows.tolist(), cols.tolist()):
        if col in outgoing[row]:
            raise ValueError("Duplicate directed edge. Fix the input condition named in the error, then rerun the operation.")
        outgoing[row].add(col)
        incoming[col].add(row)
    return outgoing, incoming


def _are_twins(neighbors: tuple[List[set], List[set]], u: int, v: int) -> bool:
    outgoing, incoming = neighbors
    return (
        (u in outgoing[u]) == (v in outgoing[v])
        and (v in outgoing[u]) == (u in outgoing[v])
        and (outgoing[u] - {u, v}) == (outgoing[v] - {u, v})
        and (incoming[u] - {u, v}) == (incoming[v] - {u, v})
    )


def _twin_partition(neighbors: tuple[List[set], List[set]]) -> np.ndarray:
    outgoing, incoming = neighbors
    n_rec = len(outgoing)
    degree_reps: Dict[Any, List[int]] = {}
    class_of = np.full(n_rec, -1)
    for u in range(n_rec):
        reps = degree_reps.setdefault((len(outgoing[u]), len(incoming[u])), [])
        rep = next((r for r in reps if _are_twins(neighbors, u, r)), None)
        class_of[u] = class_of[rep] if rep is not None else u
        if rep is None:
            reps.append(u)
    return class_of


def _refine_roles(neighbors: tuple[List[set], List[set]]) -> np.ndarray:
    outgoing, incoming = neighbors
    n_rec = len(outgoing)
    colors = np.zeros(n_rec, dtype=int)
    while True:
        ids: Dict[tuple, int] = {}
        signatures = [
            (
                int(colors[u]),
                tuple(sorted(colors[list(outgoing[u])].tolist())),
                tuple(sorted(colors[list(incoming[u])].tolist())),
            )
            for u in range(n_rec)
        ]
        refined = np.array([ids.setdefault(s, len(ids)) for s in signatures])
        if len(ids) == len(set(colors.tolist())):
            return refined
        colors = refined


def _composition_factors(k: int) -> List[str]:
    ladder = {2: ["Z2"], 3: ["Z3", "Z2"], 4: ["Z2", "Z2", "Z3", "Z2"]}
    return ladder.get(k, [f"A{k}", "Z2"]) if k >= 2 else []


def _describe_symmetry(class_sizes: List[int]) -> tuple:
    import math

    log10_order = sum(math.lgamma(k + 1) for k in class_sizes) / math.log(10)
    factors = Counter(f for k in class_sizes for f in _composition_factors(k))
    cyclic = [
        f"{name}^{count}" if count > 1 else name
        for name, count in sorted(factors.items())
        if name.startswith("Z")
    ]
    alternating = sorted(
        (f for f in factors if f.startswith("A")),
        key=lambda f: int(f[1:]),
        reverse=True,
    )
    return log10_order, " . ".join(alternating + cyclic) or "1 (trivial)"


def _neuron_label_profiles(
    n_rec: int, rows: np.ndarray, cols: np.ndarray, attribution: np.ndarray
) -> List[frozenset]:
    """Return the attribution-label set on each neuron's incident edges.

    Isolated neurons have an empty set.
    """
    if attribution.ndim != 1 or attribution.shape != rows.shape:
        raise ValueError("Attribution must align with edge endpoints. Align Attribution with edge endpoints.")
    if not np.issubdtype(attribution.dtype, np.integer):
        raise ValueError("Attribution labels must be integers. Set Attribution labels to integers.")
    profile = [set() for _ in range(n_rec)]
    for row, col, label in zip(rows.tolist(), cols.tolist(), attribution.tolist()):
        profile[row].add(label)
        profile[col].add(label)
    return [frozenset(labels) for labels in profile]


def _orbit_attribution_split(
    class_of: np.ndarray, label_profiles: List[frozenset], degree: np.ndarray
) -> tuple:
    """(Pure, mixed) counts among wired orbits of size >= 2."""
    pure = mixed = 0
    for orbit, size in Counter(class_of.tolist()).items():
        if size < 2:
            continue
        members = np.flatnonzero(class_of == orbit)
        wired = members[degree[members] > 0]
        if wired.size < 2:
            continue
        profiles = [label_profiles[index] for index in wired.tolist()]
        if all(len(profile) == 1 and profile == profiles[0] for profile in profiles):
            pure += 1
        else:
            mixed += 1
    return pure, mixed


def _task_pair_overlap(task_mass: np.ndarray, attribution: np.ndarray, n_tasks: int) -> np.ndarray:
    """Counts of shared synapses by their two largest task-mass contributors.

    Restricted to edges labeled ``shared`` (no task cleared the attribution
    threshold) that also carry nonzero gradient mass, so idle/never-trained
    edges (trivially "shared" by having no mass at all) don't pollute the
    count.
    """
    if n_tasks < 2:
        raise ValueError("Task-pair overlap requires at least two tasks. Provide the required value for Task-pair overlap.")
    if task_mass.ndim != 2 or task_mass.shape[0] != n_tasks:
        raise ValueError("task_mass must have shape (n_tasks, n_edges). Ensure task_mass has shape (n_tasks, n_edges).")
    if attribution.ndim != 1 or task_mass.shape[1] != attribution.size:
        raise ValueError("task_mass and attribution must align by edge. Align task_mass and attribution by edge.")
    if not np.all(np.isfinite(task_mass)) or np.any(task_mass < 0):
        raise ValueError("task_mass must contain finite nonnegative values. Add finite nonnegative values to task_mass.")
    if not np.issubdtype(attribution.dtype, np.integer):
        raise ValueError("Attribution labels must be integers. Set Attribution labels to integers.")
    if np.any(attribution < 0) or np.any(attribution > n_tasks):
        raise ValueError("Attribution labels must be task indices or shared. Set Attribution labels to task indices or shared.")
    matrix = np.zeros((n_tasks, n_tasks), dtype=int)
    shared = (attribution == n_tasks) & (task_mass.sum(axis=0) > 0)
    for mass in task_mass[:, shared].T:
        i, j = sorted(np.argsort(mass)[-2:].tolist())
        matrix[i, j] += 1
        matrix[j, i] += 1
    return matrix


def _print_task_pair_overlap(matrix: np.ndarray, names: List[str]) -> None:
    total = int(matrix.sum() / 2)
    print(f"  shared-synapse task pairs ({total} synapses, top-2 contributors each):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if matrix[i, j]:
                print(f"    {names[i]} + {names[j]}: {matrix[i, j]}")


def _symmetry_report(name: str, arm: Dict[str, Any], n_rec: int, names: List[str]) -> None:
    rows, cols, attribution = arm["rows"], arm["cols"], arm["attribution"]
    neighbors = _validate_topology(n_rec, rows, cols)
    outgoing, incoming = neighbors
    degree = np.array(
        [len(outgoing[index]) + len(incoming[index]) for index in range(n_rec)]
    )

    class_of = _twin_partition(neighbors)
    sizes = sorted(Counter(class_of.tolist()).values(), reverse=True)
    log10_order, factors = _describe_symmetry(sizes)
    refined = _refine_roles(neighbors)
    upper_log10, _ = _describe_symmetry(list(Counter(refined.tolist()).values()))

    label_profiles = _neuron_label_profiles(n_rec, rows, cols, attribution)
    pure, mixed = _orbit_attribution_split(class_of, label_profiles, degree)

    label_names = names + ["shared"]
    print(f"\n[19-cfsg] arm={name}")
    print(
        f"  orbits {len(sizes):4d} (largest {sizes[0]:4d})  "
        f"|Sym| ~ 10^{log10_order:.1f} <= |Aut| <= 10^{upper_log10:.1f}"
    )
    print(f"  composition factors: {factors}")
    print(
        f"  wired orbits (size>=2): {pure} attribution-pure, {mixed} "
        f"attribution-mixed (labels: {', '.join(label_names)})"
    )
    _print_task_pair_overlap(_task_pair_overlap(arm["task_mass"], attribution, len(names)), names)


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    """Run Example 18 and report topology-only symmetry for both arms.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments forwarded unchanged to Example 18.

    Returns
    -------
    dict
        Example 18's result mapping, unchanged.
    """
    result = EX18.main(argv)
    n_rec = result["config"].n_rec
    names = result["evolve"]["trick_names"]
    _symmetry_report("evolve", result["evolve"], n_rec, names)
    _symmetry_report("control", result["control"], n_rec, names)
    return result


if __name__ == "__main__":
    main()
