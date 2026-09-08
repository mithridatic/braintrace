# Kv3 closing option and comparison guards pass targeted tests

The driver accepts an optional --kv3-closing-factor, rejects zero,
negative, NaN, and infinite values before loading source data, and assigns
and reads back the factor on all somatic Kv3 segments. Omitting it keeps
older mechanism builds usable. New reports record kv3_closing_factor;
older reports have no such field.

Temporal and spatial comparisons require equal factors whenever either
report includes the field. Changed and one-sided missing factors are
invalid. Legacy pairs without the field keep their original scope.

The targeted suite passes 95 tests. Statement coverage is 100% for the
source preparation helper (12 statements), temporal audit (53), and
spatial audit (52). This is not a whole-driver coverage claim. Cases
include invalid CLI values, changed and missing metadata, individual
spike errors, missing events, malformed arrays, and source identity.

The fixed-voltage check has passed. A factor-one whole-cell equivalence
run is now required to verify the actual assignment and unchanged
response in the modified build. Driver unit checks alone do not prove
that equivalence or the physiological recovery prediction.
