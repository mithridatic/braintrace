# Grouped contraction scans

Keep the full 104-cell 30-second evolution objective and all numerical settings.
The compact contraction solver remains opt-in because its peak host memory is
above the production sample. Reduce padded schedule constants by grouping
consecutive stages into three width classes relative to the largest stage:
above one quarter, above one sixty-fourth, and the remainder. Never reorder
stages. Each group uses one forward and reverse scan, retaining eliminated rows
in existing arrays. This balances code size against padding; no model changes.

Run dense/derivative and existing physical learning parity gates with xdist,
then a bounded real four-cell learning profile and explicit production schedule
validation/cache ownership. The provisional lower-single-sample RAM gate is
revised after qualification: a 5.3x warm-learning speedup with a 0.5% higher
observed peak merits activation, while RAM reduction remains an unmet objective.
Do not claim that the observed peak decreased or that full ARC quality passed.

Production integration must validate one rooted tree with root zero, reject
duplicate children, out-of-range indices, cycles and disconnected components,
and handle a single-node tree. Cache only NumPy schedules, with weak edge-array
ownership, content checks for mutation, and a 32 MiB aggregate cap. Oversized
schedules remain usable but are not retained in the cache. Use the existing
implicit linear solve and transpose callbacks; preserve the reference path
when acceleration is disabled or static NumPy topology is unavailable.
