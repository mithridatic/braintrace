# Initial second-connection attempt: protocol disagreement

The original six-command registration stopped before connection-strength fitting
and before validation EPSP scoring. Its [terminal receipt](terminal.json) records
exit 1 after 1.230086 seconds under the 120-second cap; the original
[traceback](stderr.txt), [driver](analyze.py) and [registration](../../../specs/2026-09-14-h01-human-second-connection.md)
remain unchanged.

Source 0077 has four train spikes plus one recovery spike in calibration sweeps
0-14. The former local command adapter reconstructs a truncated fifth train pulse
at indices 68125:68225, but no recorded IN3 crossing occurs there. The
[original paired samples](missing-fifth-spike.npz) and [plot](missing-fifth-spike.png)
retain the discrepant interval in sweep 0. IN3 stays close to -74.7 mV, with no
spike or charging excursion comparable to those at the full-width pulses. This
does not establish what current was delivered: no recorded current channel is
available. The Clampex behavior for a truncated pulse remains unresolved.

Correction: the adapter at commit 77fccb74 handled a final incomplete period but
also silently constructed an incomplete pulse. Boundary regressions now reject
the latter instead of claiming either truncation or omission. Its supported
complete-pulse behavior is unchanged. Original failed execution used the helper
at 77fccb74; reproducing this exact traceback requires that source version.

The [new registration](../../../specs/2026-09-14-h01-human-second-connection-r2.md)
uses only recorded spikes as synaptic inputs and retains the absent-spike interval
as a sixth observation slot. Its [separate result](../human-second-connection-r2/README.md)
rejects strength-only synaptic transfer. It does not retroactively pass this run.

Acquisition: TONOHA v1.0 file 359655, 36,007,936 bytes, SHA1
ddabe5a487523726c42237dbf2fea0fac5a519e8, SHA256
39a77cae70f141a24b2830c4eeba7a0be43ef1cb0d8fba50cb6e1a86e045d6cd.
See [download receipt](acquisition.json) and [source identity](recording.json).
The ABF timestamp crosses midnight relative to the authors' session-date label;
the within-table date/slice/cluster/response-channel join is the linkage used.
