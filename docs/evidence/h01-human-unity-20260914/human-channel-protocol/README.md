# Human channel protocol reconciliation

The published potassium recovery experiment and released model use different
conditioning depictions. Fig. 4F shows 200 pulses before the recovery interval;
Fig. S5L depicts one depolarization, and `K_recovery` in released file 326242
implements a continuous 5000-ms command at 40 mV. The effective recovery fit
cannot be installed as channel-gate rates without resolving that difference.

A separate, usable sodium validation input is now preserved: the released
Hwide voltage command and 3,200 matching normalized amplitude observations from
16 human recordings. This checkpoint contains no model prediction, parameter
fit or six-term score change.

## Potassium evidence and unresolved command

The [original Fig. 4 image](sciadv.ade3300-f4.jpg), [supplement](sciadv.ade3300_sm.pdf),
and its rendered [Fig. S5](supplement-page-7.png) and [Table S2](supplement-page-12.png)
were acquired from Europe PMC and visually inspected. The XML, original-source
hashes and archive endpoint are recorded in [acquisition.json](acquisition.json).
The primary article is [Wilbers et al., eade3300](https://pmc.ncbi.nlm.nih.gov/articles/PMC10569700/).

Fig. 4F explicitly labels the conditioning pulse group x200, 40 mV, with recovery
at -80 mV. The inspected supplement shows a single depolarizing preparation for
the model. Neither inspected material supplies the full experimental pulse-width,
frequency and measurement-window information needed here. This is evidence of a
source mismatch, not proof of which original laboratory waveform was delivered.
The original NWB command/current traces have not been located or acquired.

The figure's recovery-delay annotation is 4-8464 ms; the selected experimental
table's actual delays are 50-7538.671875 ms. The model code uses the former range.
All three sources remain distinct. The model Table S2 confirms identical fast
inactivation parameters and slow time constants in its human/mouse columns,
consistent with the already established pooled-kinetics provenance. No guessed
conditioning waveform was simulated to bridge these gaps.

## AP command and response correspondence

The original 15,095,411-byte `H_wide.csv` is retained losslessly as
[318541-H_wide.csv.gz](318541-H_wide.csv.gz), with its published SHA1 and original
SHA256 in the acquisition receipt. Three original AP-clamp scripts are preserved
and were inspected as text only. They were not executed.

The CSV has 800,752 finite samples. At the source driver's 125 kHz clock it spans
6406.008 ms. Source-prominence peak detection finds 240 AP peaks. Peak 200 occurs
at 5177.568 ms; peak 201 follows at 5330.584 ms, after a 153.016-ms gap. Calling
the entire CSV a 200-AP command would be incorrect.

[commands.npz](commands.npz) retains two explicitly identified windows:

- The released five-AP example: inclusive source times 199-307 ms, indices
  24875-38375. Exactly five peaks occur within it.
- A prospective first-200-AP input: times 0-5182 ms, indices 0-647750, including
  the complete recorded baseline. This campaign extends the source crop formula
  to AP200 and verifies 200 peaks. It is not evidence that the released five-AP
  driver executed 200 APs.

Both windows preserve absolute source times and unmodified source voltages.
The sodium command additionally retains the source's -10 mV shift. Its unused
`LJP=16` variable is not applied. The AP-clamp sodium initializer sets ena=141 mV;
Table S2's spiking model uses 53 mV. Those contexts must remain distinct.
Neither value, nor the source thermal factor, earns new human qualification here.

The [human response file](human-hwide-responses.json) selects exactly the Hwide
fields from the original sodium Fig. 2 workbook. All 16 human rows have 200
finite values; 33 mouse rows are explicitly excluded. Original row identities,
distance_pia, leakSubt and Rn are retained. All selected leakSubt fields are 0.0;
this checkpoint does not infer the flag's meaning or resolve the laboratory
subtraction procedure from that value. No physiological QC is imposed, and this
is not a sealed blind holdout. The main sodium figure uses Hnarrow, a different
command, so its trajectory must not be paired with this Hwide input.

The potassium Fig. 2 table labels its commands h/m without the sodium
wide/narrow distinction. Exact potassium command/table correspondence remains
unresolved. Its model plotting code normalizes by maximum amplitude, whereas
the sodium plotting code uses the first amplitude; these operators cannot be
silently interchanged.

## Direct visual review

The [voltage windows](source-voltage-windows.png) preserve all samples from
2 ms before to 4 ms after AP1, AP5 and AP200. Their
[direct samples](direct-voltage-windows.json) retain source indices. AP200 has a
lower peak than AP1, with a similar broad falling phase. The -10 mV sodium shift
is explicit on every sample. These are command voltages, not paired ionic-current
measurements or measured membrane tracking during voltage clamp.

The [16 human amplitude trajectories](human-hwide-trajectories.png) were inspected
individually. Responses usually fall sharply after AP1 and then show slower
changes. Some stay relatively stable, while H21.29.195.11.41.05 declines over
much of the train. H21.29.194.11.41.01 and H21.29.195.11.41.03 show much larger
AP-to-AP fluctuations than several other records. The data span substantially
different sustained levels. All observations are retained; no mean trajectory
replaces these differences, and no mechanism is inferred solely from their shape.

## Verification and next experiment

Seventeen sibling tests pass with 100% line coverage of the command/selection
helper ([tests](tests.xml), [coverage](coverage.json)). They cover exact inclusive
slicing, clock alignment, finite coverage, species selection, missing values and
header collisions. Independent checks of original workbook fields, decompressed
CSV bytes, command sample indices and source hashes are retained in
[analysis-receipt.json](analysis-receipt.json).

The next available physiological comparison is a preregistered sodium AP-command
prediction against these per-record amplitude sequences, with the source
measurement operator and temperature-transfer assumptions kept explicit. Such a
comparison can evaluate repeated-AP behavior; it will not supply the missing
original currents, prove potassium recovery kinetics, or by itself qualify the
104-cell population. The six-term completion contract remains unchanged.
