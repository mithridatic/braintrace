# Implicit-solver wheel preparation

The [wheel audit](h01-ready-104-implicit-wheel-audit.json) verifies all 108
package files byte-for-byte against Git snapshot 526d57f and the fresh target
installation. All 33 H01 modules import from that target under Python -I.
The [installed tests](h01-ready-104-implicit-wheel-tests.json) pass 33 calcium
kernel, compiled cell and current-ownership checks; kernel line coverage is 100%
and solver coverage 96%. The launcher verifies package payloads and records all
loaded production origins on completion. Test copies change only relative imports
to absolute installed-package imports; their source and transformed hashes are
retained. No dependency or main-worktree install was modified.

This prepares an experimental wheel containing the numerical repair. Full104
installed execution, population controls/refinement and physiological gates are
still pending. The currently running full104 experiment uses source code from
the same commit, not this installed target. If qualified production code changes,
rebuild and repeat this audit before final package qualification.
