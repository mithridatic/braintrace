# E gain reassessment: source audit

The frozen B3 model already contains a somatic M current. The pinned Allen fit
and the applied genomes for opening runs at sweeps 43, 50, 53 and 56 all record
`gbar_Im = 0.0003009224287506406 S/cm2` in one soma section. Their fit and
`Im.mod` hashes match the local pinned sources. The
[audit decision](h01-e-gain-source-audit.json) records each input hash and row.

The source implements `g = gbar*m`, `ik = g*(v-ek)`, initializes `m` at its
steady-state value, and evolves it using voltage-dependent alpha/beta rates
with temperature correction. Its source comment attributes the model to
Adams et al. 1982 and bullfrog sympathetic neurons. Source presence and
parameter application do not verify human-specific kinetics or establish
that changing this current would repair the gain defect.

The older reassessment's phrase "Kv7/M kinetics absent from the Allen genome"
must therefore not be read as absence of an M current. A future hypothesis
would have to specify a change to the existing mechanism or its distribution,
and predict effects on both low-drive counts and the protected higher-drive
and subthreshold responses. No such intervention was tested by this audit.

The tested Ih and leak interventions failed their registered gates. Those
finite tests and the registered additive prediction do not prove that every
passive parameterization is impossible, nor that a particular active current
is necessary. Keep these broader statements as hypotheses, not measured
causal conclusions.

The historical stage decision JSON is preserved. B3 remains unpromoted;
physiological failures, campaign limits and the sealed holdout are unchanged.
This correction arose from relying on the reassessment wording without first
checking the applied genome. Future mechanism hypotheses must include that
source check before assigning a new experimental cap.
