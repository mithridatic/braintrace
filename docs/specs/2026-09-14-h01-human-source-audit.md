# Preliminary source audit for human-only qualification

Date: 2026-09-14. Read-only source inspection, no model execution or promotion.
Related plan: [human unity](2026-09-14-h01-human-unity.md).
All local source paths below refer to the primary checkout inspected at dba5ae6
with its existing dirty files intact. Implementation was subsequently approved
by the user ("Execute the plan.").

Correction established during implementation: the historical converter's
"soma-labeled" count of 1,550 was actually H01 dendrite code 1; the true source
soma code is 3 (19 samples in the selected component). The old converter's
interpretation is invalid. See
[verified correction](../evidence/h01-human-unity-20260914/conversion-correction.json)
and [visual review](../evidence/h01-human-unity-20260914/result.md).
The byte hashes below describe the earlier checkout snapshot; the implementation
decision records current hashes and verifies the same original archive member.

## Exact failed-transfer input identified

`docs/evidence/h01-e-morphology/donor.json` identifies H01 cell 955432427,
archive member `955432427.0.swc`, member SHA256
`cc52cc1b8d396bc840648faee816f3e02b2702539c7235129625e5cbba8b9f27`.
This is a manifest hash, not a newly verified hash of the archive member.

The manifest reports 11,524 input nodes and 9,975 output nodes, with 1,550
soma-labeled nodes collapsed to a sphere of radius 6.864013671875 um from
the Allen donor. The largest original soma-labeled skeleton radius is
1.246794312 um. Neither radius alone establishes the real H01 soma size.
The conversion labels the longest subtree apical and floors radii at 0.05 um.
The reference driver replaces the axon with its donor stub. This diagnostic
must be distinguished from the deployed neutral-label SWC import path.

The published fit assigns dendritic/apical specific capacitance
2.3031548608821226 uF/cm2 and soma/axon 1.0 uF/cm2. Inspect effective
electrical load and explicit membrane geometry separately: onset capacitance
is not by itself a direct surface-area measurement.

## Public mesh access and format established

The public metadata endpoints were successfully read with Invoke-RestMethod
outside the sandbox after in-sandbox TLS credential failures. No account,
credential change, certificate bypass, or paid resource was needed.

- https://h01-release.storage.googleapis.com/data/20210601/proofread_104/mesh/info
  returns `neuroglancer_multilod_draco`, 16-bit quantized vertices, transform
  diagonal (16, 16, 33), zero translation, LOD scale multiplier 1. Its sharding
  uses murmurhash3_x86_128, preshift 6, minishard bits 8, shard bits 10;
  data and minishard indexes are gzip encoded.
- https://h01-release.storage.googleapis.com/data/20210601/proofread_104/skeletons/info
  returns `neuroglancer_skeletons`, identity transform and a scalar float32
  radius attribute. Sharding uses murmurhash3_x86_128, preshift 1,
  minishard bits 3, shard bits 2, with gzip encoding.

These are different storage formats. The SWC archive's position conversion
(32, 32, 33 nm) cannot simply be applied to mesh vertices or substituted for
the precomputed skeleton metadata. Decode LOD/fragment offsets, quantization,
and physical transforms using the specified format before comparing areas.
The metadata difference is not evidence of an importer bug.

Mesh fragment selection, component correspondence and surface-area measurement
remain unperformed. Metadata access does not close anatomy_transfer.

## Human biological provenance has a concrete failure

`braintrace/datasets/h01_l2_channels.py` explicitly describes its channel laws
as including borrowed nonhuman mechanisms. The local source used by the
stage-1 manifest is `.cache/human-pyramidal-l2/kv3-closing-source`.

| Mechanism | Local source evidence | Current conclusion |
| --- | --- | --- |
| NaTs | NaTs.mod cites Colbert and Pan 2002; the published fit includes somatic NaTs | The cited primary study reports rat neocortical layer-5 recordings. Human-specific kinetic validation remains necessary. |
| Im | Im.mod names Adams et al. 1982 and bullfrog sympathetic neurons | Nonhuman origin explicitly documented in the local source; independent human validation not established by this audit. |
| K_P, K_T | Both source headers cite Korngreen and Sakmann 2000, young rats | Nonhuman origin explicitly documented in the local source; independent human validation not established here. |
| Ih | Ih.mod cites Kole, Hallermann and Stuart 2006 | Citation identified; full experimental methods and human replacement/validation still need examination. |
| Kv3_1 | Header says Kv3-like current without a primary citation | Human kinetic provenance unavailable from the header. |
| Other included calcium, potassium and persistent-sodium mechanisms | Source references located, not fully audited | No human-only qualification granted. |

The primary sodium study is
[Colbert and Pan 2002](https://www.nature.com/articles/nn857), whose abstract
identifies rat neocortical slices. This establishes the species of that source;
it does not by itself determine whether independently measured human kinetics
could support the same equations. Do not mistake fitting human conductance
densities for this missing kinetic validation.

A potential human sodium-data lead is
[Functional properties of rat and human neocortical voltage-sensitive sodium currents](https://pubmed.ncbi.nlm.nih.gov/8201401/).
The indexed abstract reports adult human recordings as well as rat recordings.
Only the abstract was recovered; human-specific numerical data, conditions,
availability and suitability for the targeted cells are unverified. The animal
comparison cannot substitute for extracting the human measurements themselves.

## Snapshot hashes measured during this audit

| File | SHA256 |
| --- | --- |
| docs/evidence/h01-e-morphology/donor.json | 0ad796c6e7a4ef2d9fadfc78cb36606f6d1c98cc394462cd5cd34855a86344d1 |
| docs/evidence/h01-e-morphology/h01-955432427.swc | 062da7ecda0d555b4f4be3a394ffa4ac6565d0b8f86e8664062e82bb43c4a66f |
| .cache/human-pyramidal-l2/kv3-closing-source/NaTs.mod | 6ba797bada310a6b880f4ab2a86e915894cc205d411474c0ce78f806af532a1f |
| .cache/human-pyramidal-l2/kv3-closing-source/Ih.mod | 8616325d83add9a1a6546972352b4a62e57095ba25f3c6b5a57ab8b397e2d662 |
| braintrace/datasets/h01_l2_channels.py | 60408006f4e4175ae2ac9de5f2b8d7326d03c8c116715970af244dc245a99111 |

## Next actions and completion boundary

Implement the registered mesh/component area audit after approval, preserving
separate source and simulation geometry. In parallel with that work in the
ordinary task sequence, inventory human-specific measurements for the active
channel laws and all unmatched donor types. No subagents are requested here.

The current six values are unchanged. No morphology defect has yet been
demonstrated, no channel replacement is qualified, and no new physiological
run has occurred. This audit changes the next action by identifying the exact
geometry source, its mesh decoding requirements, and specific biological laws
that cannot currently support the user's human-only qualification.
