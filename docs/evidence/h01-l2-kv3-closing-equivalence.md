# Default Kv3 factor preserves the complete recorded control

All 19 saved arrays match exactly over 833375 samples. This includes
time, voltage, applied current, raw indices, and all recorded soma states
and currents. Both raw mappings are exact; input plateau errors are zero.
All values are finite, and both traces end at 2100 ms.

Physical setup is unchanged. The new default factor is one; the old
source has no factor field. Only the Kv3 source changes among eleven
mechanisms, exactly as specified by the preparation manifest. The new
library hash matches the successful fixed-voltage gate check. Remaining
metadata differences are the library hash and integration time.

The [full comparison](h01-l2-kv3-closing-equivalence.json) pins both runs
and retains each array result. This verifies the default implementation
at the frozen candidate settings. The half-closing-factor response
prediction can now be tested. No physiological qualification follows
from default equivalence.
