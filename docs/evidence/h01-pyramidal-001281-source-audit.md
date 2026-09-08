# Independent adult pyramidal recording: datum check

The pinned [DANDI release](https://dandiarchive.org/dandiset/001281/0.251110.1752)
contains human L2/3 pyramidal recordings from a study of age dependence.
The first listed asset identifies a 62-year-old participant. Its description
states whole-cell current clamp with an I-V protocol. This is an independent
reference candidate, not the H01 donor or the Wilbers recording.

The downloaded file's SHA-256 matches the archive digest. It contains
12 voltage/current pairs with matching sample counts, start times, and
100 kHz sampling. However, all voltage series declare a conversion factor
of 1e-12 to volts. The first stored voltage range is about -0.091 to
-0.067, giving about -9.1e-14 to -6.7e-14 volts under the declared conversion.

The stored values resemble membrane voltages expressed directly in volts.
That is an inference, not an accepted correction. Do not silently remove
the declared conversion or use this recording as a calibration target.
Verify the original conversion code or an independent source datum first.
The current series uses a conversion factor of 1 to amperes.

The [source audit](h01-pyramidal-001281-source-audit.json) retains the pinned
asset ID, digest, every paired-series range, and the qualification boundary.
Cached files are under `.cache/human-pyramidal-001281`. The archive search
for the candidate Wilbers filename did not produce a raw-file link; this
does not prove that the recording is unavailable elsewhere.
