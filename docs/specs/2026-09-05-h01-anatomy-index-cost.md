# H01 anatomy index cost

The live measured-pair construction stack enters `_geometry_signature` from
region validation during CV construction. BrainCell `MorphoBranch.index`
rebuilds the complete morphology index map on every call. The adapter calls it
for each branch, edge endpoint, and many source segments.

Use the public default branch ordering once per traversal. Map edge node
objects to those indices. Do not cache a geometry fingerprint across edits.
Keep the fingerprint bytes, source-node locations, regions, and projections
unchanged. Do not change channels, cable properties, or integration settings.

Before implementation, reproduce repeated index-map calls with a small
branched fixture. Compare the resulting fingerprint to the original algorithm.
Retain all wrong-geometry rejection tests. Check a real H01 component after the
fixture passes. Timing is diagnostic; signature and location equality decide
whether the change preserves behavior.
