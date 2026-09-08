# Slower opening improves the rising phase but lowers spike peaks too far

The predefined rising-phase prediction passes. Five complete positive-peak
events remain. Absolute rise-to-peak errors decrease by 0.061148,
0.061771, 0.061973, 0.062182, and 0.061612 ms. Each change exceeds
the required 0.01 ms diagnostic margin. Rising phases remain too short.

| Spike | Onset error (ms) | Peak error (mV) | Duration error (ms) |
| --- | --- | --- | --- |
| 1 | +6.175204 | -6.254857 | -0.017173 |
| 2 | +2.813565 | -6.027850 | -0.220253 |
| 3 | +25.801867 | -6.212811 | -0.065043 |
| 4 | +29.467141 | -6.135621 | -0.062167 |
| 5 | +51.824002 | -5.933022 | -0.054580 |

Errors are model minus human. Duration is time above -20 mV. Interval
errors are -3.361639, +22.988302, +3.665273, and +22.356862 ms.
Recovery-minimum voltage errors are +1.130506, -1.726680, +1.348920,
and +1.387943 mV. Three minima are now too shallow; the second remains
too deep. Passing the rising-phase prediction does not accept the cell.

Only the opening factor changes among physical setup fields, from one
to two. Source and library hashes are identical between these runs.
Raw mappings are exact. Both recorded inputs have zero plateau error,
excluding 1e-7 ms around command transitions. Finite traces end at 2100 ms.
The [full result](h01-l2-sodium-opening-result.json) retains all phases,
events, intervals, minima, residuals, and source hashes.

The gate-closing law remains fixed, but the cell voltage trajectory and
other currents respond to the altered opening. It is therefore incorrect
to infer unchanged falling or recovery behavior from that fixed law.
Candidate-specific numerical and subthreshold checks remain open. No
unique human channel explanation or physiological qualification follows.
