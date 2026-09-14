# Avoid discarded H01 constructor discretization

The 30-second full evolution goal remains open. cProfile on four real cells
shows 14.6 seconds spent constructing a default preview that painting discards.
H01Cell should perform eager policy and geometry validation in its constructor,
but defer CV mechanism and node-tree assembly until a caller requests a public
preview or initializes the cell. Keep ordinary public cvs behavior, default paint
rules, numerical settings, model geometry, and state initialization unchanged.

Use a constructor-scoped flag and override cvs only during the parent constructor's
discarded validation access. Resolve bounds with the default paint rules and build
the existing validated geometry, whose owner-scoped cache serves initialization.
Always clear the flag on constructor failure. No placeholder may escape construction.

Regression tests must prove no node-tree construction during declaration, ordinary
preview construction afterward, policy failure propagation, and initialized parity.
Run existing geometry, cell, solver and learning tests with xdist on Vast. Measure
the bounded four-cell construction/init profile before making any runtime claim.
