# Explicit opening factor and audit invariants

The reference driver accepts --sodium-opening-factor as an optional
positive finite value. When specified, it assigns and reads back the
factor on each somatic NaTs segment. The run metadata record the value.
An omitted setting leaves older mechanisms usable and records null.
The modified mechanism itself defaults to one.

Temporal and spatial audits reject unequal opening factors or a field
present in only one run. Legacy pairs without this field retain their
original comparison scope. Same-factor pairs remain usable.

Ten new checks failed before implementation: four CLI cases and six
changed or missing factor cases. After the driver and audit update,
83 targeted tests pass. The two audit modules and source-patch helper
have 100 percent statement coverage. This does not claim full driver
coverage or whole-cell equivalence.

The CLI test previously accepted any exit code 2, which could hide an
unrecognized flag. It now also rejects the unrecognized-argument error.
Future setting tests must distinguish parser recognition from value
validation, and new physical settings must enter the numerical invariants.

Whole-cell default-factor equivalence remains required. The ongoing
midpoint experiment uses the earlier mechanism and has not been changed.
No opening-speed intervention has been applied to a full cell.
