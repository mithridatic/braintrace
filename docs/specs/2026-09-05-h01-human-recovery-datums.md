# Direct human recovery datums

Extend the existing calibration event targets with directly sampled recovery
minima. Use E sweep 50 and I active traces at 0.19 and 0.27 nA. Do not read
E sweep 53 or derive new measurements from reserved I 0.23 nA.

For each complete event, use samples after its falling -20 mV crossing and
before the next rising crossing. For the last event, end the window at pulse
end and mark it terminal. Record the minimum sample voltage, its time, delay
from the preceding peak, and both window boundaries. If the minimum is at a
window edge, flag it; do not silently call it a resolved recovery minimum.
This is a sampled-window datum, not a causal channel diagnosis or an AHP
amplitude relative to an inferred threshold. Retain corrected voltage and
source clocks without alignment. Keep tolerances unresolved. Verify source
hashes where the existing datum pins the same raw file.
