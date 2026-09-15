# Human channel protocol reconciliation and AP-command inputs

Continue the approved human-only campaign without promoting any ledger term.
The preceding effective potassium recovery fit is not a channel-gate fit.

Preserve the Europe PMC full-text XML and supplementary archive references,
original Fig. 4 image and supplement PDF, with byte hashes. Inspect Fig. 4F,
Fig. S5L and Table S2 visually. Distinguish the experimental figure's 200-pulse
conditioning depiction from the single depolarization in the model supplement
and released K_recovery driver. Record incomplete experimental timing as unknown;
do not infer pulse width or frequency from an unrelated AP experiment. Likewise,
keep the figure's 4-8464 ms delay annotation distinct from the workbook's actual
50-7538.671875 ms observations. No guessed-command simulation or gate fit.

Acquire the original H_wide.csv and AP-clamp scripts from the already pinned
Dataverse release 3.0. Verify published byte lengths and SHA1 and retain SHA256.
Inspect scripts as text only. Compress the original CSV losslessly if useful,
checking its decompressed bytes against the original hash. This is a recorded
voltage-waveform input used by the published simulator; it is not the missing
experimental recovery command or an acquired current trace.

Verify every voltage sample is finite and one-dimensional. The released driver
sets 125 samples/ms (125 kHz) and takes an inclusive 199-307 ms crop for its five
AP example. Preserve that crop exactly, with absolute sample indices and time.
Retain the full command and detect all peaks using the source prominence 50 mV.
Do not silently label the full CSV as a 200-AP trace: it has 240 detected peaks,
including a second segment after a gap. A separately named prospective 200-AP
input may select original times 0-5182 ms, retaining the full recorded baseline;
the 5182-ms endpoint extends the source crop formula to AP200. Verify exactly
200 peaks and full source sample coverage. This extended window is this campaign's
preparation, not a claim that the released five-AP driver executed 200 APs.

For the sodium command retain the source's -10 mV offset separately from the
unshifted waveform. Do not apply the unused LJP=16 variable as another shift.
The source sodium AP driver initializes ena=141 mV whereas Table S2's spiking
model lists 53 mV; keep those experimental/model contexts separate. No source
reversal or thermal parameter is newly qualified by this acquisition.

Extract only identified human response records from the Na Fig. 2 workbook's
relative_amp_Hwide fields, preserving all 200 AP values including missing data,
source row, Filename, distance_pia, leakSubt and Rn. Normalize header spacing only
for field lookup and reject duplicate or incomplete 1-200 header sets. Record
all-missing response rows separately, never replace missing responses with zeros.
The Hnarrow fields used for the main sodium figure are a different waveform and
must not be paired with H_wide.csv. This selection does not perform physiological
QC or establish a blind holdout. Preserve human identity by the authors' H/M
filename convention. Unknown or duplicated identities fail closed.

The potassium Fig. 2 table labels input only h/m, without the sodium wide/narrow
distinction. Do not claim exact K command/table correspondence until independently
resolved. Record the source potassium normalization by maximum versus sodium by
first amplitude as different measurement operators. No model predictions are
generated in this input checkpoint. Tests cover exact inclusive source slicing,
nonfinite/out-of-range commands, species rejection, missing responses and header
collisions. Plot and inspect the retained voltage windows and human amplitude
trajectories, retaining all samples and identities in artifacts.
