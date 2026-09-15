# First population geometry attempt: terminal comparison failure

This attempt completed source-edge and CV geometry checks for 15 cells, then
terminated with exit code 1 after 26.153 seconds. It did not time out. Cell
1830470325's reconstructed soma-region boundaries failed the 1e-12 normalized
comparison against the pinned construction reference. The log, terminal receipt,
executed source hashes and 15 completed records are preserved.

The [second attempt](../population-geometry-r2/README.md) retains the same
comparison tolerance and collects its outcome separately from geometry
conservation, so all 104 cells can be inspected. Nothing in this first prefix is
an all-population pass. Its plotting script was prepared but not run because
the required complete result was absent.
