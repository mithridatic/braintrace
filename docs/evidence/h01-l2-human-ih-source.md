# Human HCN model source identifies two separate assumptions

The [Kalmbach et al. study](https://doi.org/10.1016/j.neuron.2018.10.012)
reports HCN-related differences in human supragranular pyramidal neurons.
Its [released model](https://github.com/AllenInstitute/human_neuron_Ih/tree/a5af6f349e1ef269f693aa923a3e698593edb0c7)
provides an inspectable basis for further diagnostics, not proof that our
reference cell has the same parameters.

The repository is pinned at a5af6f349e1ef269f693aa923a3e698593edb0c7.
Its README links the study. The uniform-example node table selects
478230220.json and morphology 478230220.swc. This differs from our
reference specimen 541563728 and from H01 anatomy. The
[manifest and equation comparison](h01-l2-human-ih-source.json) record
file hashes, source paths, and numerical rate evaluations. No downloaded
source code was executed or copied into production modules.

The selected fit assigns Ih_mod to soma, apic, and dend, with densities
0.000100314, 0.000100549, and 0.000100185 S/cm2. Our current fit assigns
Ih only to soma. The current reconstruction has about 592 um2 of soma
and 14,846 um2 of dendritic membrane. Adding channels at the same density
to those dendrites would change both location and total conductance;
it must not be described as a pure relocation experiment.

The released mechanism replaces v with v-20 only in the activation-rate
alpha expression. Beta remains unchanged. This changes both equilibrium
availability and the relaxation time; it is not a uniform translation of
the complete gate law. The mechanism cites a prior non-human formulation.
Treat the modified law as a published model assumption, not a direct
human channel measurement.

The next topology test should preserve total maximum Ih conductance
while changing its distribution, with original kinetics fixed. A later
density or rate change must be a separate intervention. Source model
passive fits and other-cell density values are not automatically valid
for the current reference. No current candidate has been promoted.
