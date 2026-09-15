# Initial synaptic run stopped before fitting

The terminal process exited 1 after 0.883 s when calibration sweep 2 contained
seven upward zero crossings instead of the assumed six. No optimizer or
validation comparison ran. The original [executed specification](executed-spec.md),
runner, stderr and terminal receipt remain unchanged.

The [direct comparison](unexpected-crossing.png) and [original samples](unexpected-crossing.npz)
show one full fifth spike with a small falling-phase recrossing near 0 mV.
They also exposed an incorrect zero command during that spike in pyabf's
reconstruction. The source Pulse epoch contains a final pulse that the parser's
floor(duration/period) calculation omits. The command panel in this initial
figure therefore shows the defective parser output, not a verified laboratory
command. The [corrected run](../human-synaptic-prediction-r2/README.md) preserves
this result, repairs the local reconstruction, and retains the extra crossing.
