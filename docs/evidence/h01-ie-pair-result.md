# I-to-E pair with the literature receptor: result (W4)

Runs: `h01_ie_pair_response.py run --hold-na 0.4` (three 40 ms runs of about 170 s
each through the closing-restored wrapper), scored into
[h01-ie-pair-response.md](h01-ie-pair-response.md). Prediction registered in the
spec before the runs: with the literature conductance the somatic IPSP at a
depolarised hold is below 0.1 mV and the receptor-site response is 10 to 100 times
larger; the pair's deficit is the electrotonic distance of the receptor site, not
the conductance. Rejection: a somatic IPSP above 0.5 mV.

## What was measured

E held with 0.4 nA from 0 ms (the cell is still depolarising during the run: soma
−71 mV at the I spike, −60 mV at 40 ms; receptor site 7 mV below the soma
throughout). One I spike at 14.78 ms in every run. Connected minus disconnected:

| Receptor | Conductance | Reversal | Decay | Site (mV) | Soma (mV) | Soma / site | Latency | Decay at site |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| literature (Szegedi 2020, 2017) | 3.1 nS | −75 | 4.18 ms | +0.86 | +0.008 | 1/110 | 0.5 ms (the delay) | 4.4 ms |
| assumed (circuit default) | 20 nS | −80 | 5 ms | −5.2 by 40 ms | −0.38 by 40 ms | 1/14 | 0.5 ms | none within the window |

The literature response has the receptor's own kinetics (latency equals the
transmission delay, decay 4.4 against 4.18 ms) and is **depolarising**: the site
sits at −77 to −78 mV, below the −75 mV reversal, while the soma is at −71. The
assumed response is not an IPSP shape: the 20 nS shunt at −80 mV pins the site near
−79 mV while the control keeps depolarising, so the difference grows for the whole
window. Its ratio of peak soma and peak site differences (1/14) is larger than
the literature condition's (1/110), but those peaks occur at different times.
These ratios are descriptive; they are not matched transfer-impedance estimates.

## Reading in the energetic frame

The receptor is a Norton source (conductance transient with reversal) feeding the
E load at the receptor site. What reaches the soma is that source seen through the
dendritic Transmit path: a 110-fold attenuation for a small conductance. A
human-sized unitary conductance moved this E soma by about 0.008 mV in the
tested drifting state at a reversal of -75 mV. This does not bound the response
at other reversals or membrane states. The site-to-soma difference is consistent
with strong attenuation, but the runs change conductance, reversal and decay
together and do not isolate placement as the sole cause.

## Prediction scored

- Somatic response magnitude below 0.1 mV: **held** (0.008 mV); predicted IPSP
  polarity was not observed (the response is depolarising).
- Site response 10 to 100 times the soma's: **outside the registered band** (111.3 times).
- Rejection (somatic IPSP above 0.5 mV): not met.

Audit: both connected and disconnected E event lists are empty. These runs
cannot score spike suppression or E cycle-length change. Site latency is 0.5 ms;
soma latency to 10 percent is 2.38 ms. The wrapper still restores I sodium
closing kinetics, so this is diagnostic pair evidence, not a qualified E/I pair.

## What follows

1. The synapse defaults in `h01_ei_circuit.py` stay at the assumed values until the
   placement question is answered; the literature values are exposed as flags and
   pinned in `h01-ie-synapse-literature.json`.
2. The placement question is anatomical: contact 8105899's postsynaptic endpoint is
   0.13 µm from the E cable at `cable_location [2805, 0.93]`, a dendritic site. Whether
   a PV basket contact on a human L2/3 pyramid should sit perisomatically is a
   question for the anatomy (the 5-voxel re-check confirms the endpoint) and for
   the E model's dendritic attenuation (its dendrites carry cm 2.303 and passive
   membrane; the axial load already explained the rise-rate fault). Both are
   returned to the user.
3. The E cell rests at −84 mV, 12 mV below the human's −72. These drifting runs
   do not establish the receptor-site voltage or response at a steady human-like
   rest. Resting-state correction and a matched-state synaptic test remain open.
