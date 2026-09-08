H01 human neurons in BrainCell
=============================

``braintrace.datasets.h01`` imports the released H01 human cortical neuron
geometry into BrainCell. It is an installable library module, independent of
ARC, Example 21, and any learning algorithm. Nothing downloads at import time.

Install the version of BrainTrace containing this adapter in the target
project's environment, for example ``python -m pip install /path/to/this/checkout``
or install its built wheel. These instructions do not imply that this feature
has been published to PyPI. BrainCell must provide ``Morphology.from_swc`` and
the multicompartment ``Cell`` API; both were verified in the local 0.1.0 runtime.

Download once and reuse offline
-------------------------------

.. code-block:: python

   from braintrace.datasets.h01 import fetch_h01, H01Archive

   archive = fetch_h01("./data/h01")  # explicit download, about 60 MB
   print(archive.neuron_ids)         # original IDs for 104 cells
   print(archive.components("810151953"))  # (0,)
   imported = archive.load("810151953", component=0)

   # Subsequent projects can share the downloaded archive without network access.
   archive = H01Archive("./data/h01/104_proofread_neurons_swc.zip")

The adapter pins the SHA-256 of the official archive. Corrupt caches raise an
error. Downloads are staged before becoming a cache entry. The original ZIP is
preserved, and archive entries are read without extracting arbitrary paths.

Construct a BrainCell cell
-------------------------

.. code-block:: python

   import braincell
   from braintrace.datasets.h01_demo import make_passive_cell
   import brainunit as u

   # Configure a new model using the imported geometry.
   cell = braincell.Cell(imported.morphology, cv_policy=braincell.CVPerBranch())
   # Add the desired membrane properties, ion channels, synapses and probes
   # before initializing this cell.

   # Or run the explicitly hypothetical passive demonstration:
   cell = make_passive_cell(imported, current_na=0.001, duration_ms=1.0)
   result = cell.run(dt=0.025 * u.ms, duration=1.0 * u.ms)
   voltage = result.traces["voltage"]
   print(imported.provenance)

The passive example uses uniform capacitance of 1 uF/cm2, axial resistivity
of 100 ohm cm, leak conductance of 0.1 mS/cm2, and resting/leak potential
of -65 mV. These are demonstration assumptions, not measurements or a fit
to the imported human neuron. The clamp and probe are attached at the source
tree root, which is not necessarily the cell body. ``Cell.run`` uses a compiled
``brainstate.transform.for_loop`` internally.

The same example is available from any directory after installation:

.. code-block:: shell

   python -m braintrace.datasets.h01_demo --download ./data/h01 --output ./h01-run.json
   python -m braintrace.datasets.h01_demo --archive ./data/h01/104_proofread_neurons_swc.zip --neuron 810151953 --component 0

What is preserved and what is assumed
------------------------------------

The official archive contains **3,327 connected components belonging to 104
cells**, not one ready-to-run SWC per complete neuron. ``components`` lists all
of them. ``load`` requires an explicit component ID; it never joins components
or silently chooses the largest. A single-point component is available in the
source archive but cannot define a cable and is rejected by ``load``. Cell
``810151953`` has one archived component, which makes it a convenient example.
This does not establish that its arbor is complete outside the imaged tissue.

The archive's seven columns are not standard simulation-ready SWC data:

* Positions use 32 x 32 x 33 nm skeleton voxels. The adapter converts them to
  micrometers with factors ``(0.032, 0.032, 0.033)``.
* Radii are already in nanometers, so their conversion factor is ``0.001``.
* Node annotation codes use H01's vocabulary, not standard SWC types. In
  particular, code 1 denotes dendrite, whereas ordinary SWC uses 1 for soma.
  Most source samples have the unclassified code -1.

To avoid assigning false biological identities or inventing classifications,
the simulation morphology uses neutral ``custom`` branches. All original
annotation codes, IDs, positions, radii and parents remain in the read-only
``imported.source_rows`` array. H01's analysis code documents codes 0 (axon),
1 (dendrite), 2 (astrocyte), 3 (soma), 4 (cilium), 5 (axon initial segment),
and 1000-series myelin annotations. They are predictions/annotations, not a
uniformly manually verified cell-compartment classification. The optional
anatomy adapter below turns them into explicitly bounded selections.

``imported.normalized_swc`` contains the converted, renumbered SWC. Parent
relationships and absolute physical coordinates are retained. BrainCell reads
it in ``neuromorpho`` mode to retain parent attachment geometry. Its diagnostic
report is returned in ``imported.report``. The expected ``no_soma_samples``
diagnostic reflects the neutral labeling policy, not removal of measured soma
geometry. Invalid radii, non-finite data, missing parents, duplicate IDs and
cyclic/disconnected trees are rejected before BrainCell can invent repairs.

Use with PP-Prop or another example
---------------------------------

Import the adapter from the installed package; do not import Example 21 to
obtain a neuron. Keep the archive and ``imported.provenance`` alongside the
experiment's configuration. Construct a fresh ``braincell.Cell`` for each
independent stateful model and supply the experiment's own sensory encoder,
synapses, electrical parameters and readout. A tuple of cell IDs/component IDs
can select multiple geometries; their identities and disconnected components
remain explicit.

Replacing Example 21's single-compartment HH cell requires adapting its state
ownership, input-current injection and readout to the multicompartment cell.
The anatomical import and forward run do not establish PP-Prop eligibility
trace correctness. Qualify that learning path against a finite-window oracle
before making gradient or training claims. Example 21 is unchanged here.

Synaptic partner connectivity, synaptic strength, ion-channel kinetics, learning
rules and missing external inputs are not supplied by this morphology adapter.
The output is a measured anatomical reference with caller-selected dynamics,
not an emulation of a complete human cortical circuit.

Use anatomical labels for stimulation, recording and channels
------------------------------------------------------------

.. code-block:: python

   from braincell.mech import Channel
   import brainunit as u

   anatomy = imported.anatomy()
   print(anatomy.label_counts)
   soma = anatomy.soma_location()
   dendrite = anatomy.region("dendrite")
   cell = make_passive_cell(imported, recording_location=soma,
                            stimulus_location=soma)
   # Caller-chosen demonstration conductance; not supplied by H01.
   cell.paint(dendrite, Channel("IL", g_max=.2 * u.mS / u.cm**2, E=-65 * u.mV))

``soma_location()`` selects the largest-radius sample actually labelled soma
(ties use the smallest original node ID). It raises if the selected component
has no such sample. For example, ``546925828.0`` lacks soma annotations, while
``546925828.1`` contains them. A source tree root is not a substitute for soma.

``anatomy.location(original_node_id)`` selects a particular original SWC sample.
``anatomy.locations("axon_initial_segment")`` selects all samples bearing that
label; a known but absent label produces an empty selection. Unknown numeric
codes survive as ``unknown_<code>``; typos in label names raise.

Regions have two explicit policies:

* ``strict`` (default): a cable segment is selected only when both endpoints
  carry the requested label. It leaves mixed-label edges unselected.
* ``sample_neighborhood``: each sample's label extends halfway along its
  incident cable edges. This is an inferred spatial boundary. It does not
  propagate a known label through unclassified samples or across branches.

.. code-block:: python

   inferred_dendrite = anatomy.region("dendrite", policy="sample_neighborhood")
   print(inferred_dendrite.policy)

The labels are sparse: in ``810151953.0``, 11422 of 14215 samples are
unclassified. There are only 40 edges with dendrite labels at both endpoints.
Sample counts are not percentages of cable length. Keep the selected policy
in experiment provenance; neither policy establishes verified boundaries.
BrainCell discretization determines how these continuous intervals resolve
onto electrical compartments. Refine the discretization when boundaries matter.

Selections check geometry and topology when evaluated. They accept BrainCell's
internal morphology clone, but reject different or modified geometry. Rebuild
``anatomy`` after changing a morphology. Coincident source coordinates are
rejected by the anatomy adapter because source identity would be ambiguous;
this restriction does not change the basic SWC loader.

Load cell type, layer, and aggregate annotations
----------------------------------------------

.. code-block:: python

   from braintrace.datasets.h01_annotations import fetch_h01_annotations, H01Annotations

   annotations = fetch_h01_annotations("./data/h01")  # explicit ~19 MB download
   # Offline reuse: annotations = H01Annotations("./data/h01")
   layer2_pyramidal_ids = annotations.select("L2", "pyramidal")
   metadata = annotations.metadata(imported.neuron_id)
   print(metadata.tags)
   print(dict(metadata.measurements))
   print(dict(metadata.descriptions))

The release's segment-property file directly maps all 104 imported cell IDs to
tags, including cortical layer, pyramidal/interneuron classifications, and
descriptive qualifiers. Preserve qualifiers such as ``possible-interneuron``.
Tags are anatomical annotations, not transcriptomic or electrical cell types.
``select`` requires all requested tags and rejects unknown tag names.

Numeric properties retain their source names and descriptions, including
volume in 8 x 8 x 33 nm voxels (``NVx``), incoming/outgoing counts (``NSI``,
``NSO``), and incoming excitatory/inhibitory counts (``NSIe``, ``NSIi``).
These describe the released cell, not just the component loaded into BrainCell.
For ``810151953`` the source tags are ``L2, pyramidal, neuron`` and the
incoming totals are 388 = 138 excitatory + 250 inhibitory.

Use observed synapse positions
-----------------------------

.. code-block:: python

   rows = annotations.synapses(imported.neuron_id, role="post")
   projections = anatomy.project_synapses(rows, max_distance_um=2.0)
   # Each result stays paired with its source row, including rejections.
   accepted = [(row, p) for row, p in zip(rows, projections) if p.status == "projected"]
   row, projection = accepted[0]  # Check for an empty result in your application.
   print(row.source_row, projection.distance_um)
   cell = make_passive_cell(imported, stimulus_location=projection.location,
                            recording_location=anatomy.soma_location())

The CSV contains 264090 rows. ``source_row`` identifies a line in the pinned
file, not a biological synapse ID. ``role`` says whether the named cell is
presynaptic or postsynaptic. ``center_um``, ``pre_um``, and ``post_um`` retain
all supplied positions; ``position_um`` chooses that cell's endpoint. They
use **8 x 8 x 33 nm** voxels, distinct from the skeleton's 32 x 32 x 33 nm grid.

Projection checks cell identity first and uses nearest points on cable
segments, not merely the nearest skeleton sample. Supply an explicit distance
threshold. Every row produces ``projected``, ``too_far``, or ``ambiguous``, plus
distance evidence. Rejected rows have no placement selection. Incident edges
at one shared source node resolve to that node; equal-distance distinct sites
remain ambiguous. Spatial proximity alone does not prove component membership.

The CSV has **no individual E/I labels, partner neuron IDs, synapse IDs, or
conductances**. Cell ``810151953`` has 810 post rows and 44 pre rows in this CSV,
which differ from the segment property's 388 incoming and 4 outgoing counts.
The adapter preserves this discrepancy. It does not deduplicate, assign
receptors from aggregate counts, or claim the rows are individually proofread.
``annotations.synapse_provenance`` records these limits and the source hash.

Installed annotated demonstration
----------------------------------

.. code-block:: shell

   python -m braintrace.datasets.h01_annotated_demo --archive ./data/h01/104_proofread_neurons_swc.zip --annotations ./data/h01 --max-distance-um 2 --output ./annotated-run.json

This runs 40 compiled steps in 64-bit precision, records at the labelled soma, applies a current
clamp at the closest accepted postsynaptic endpoint, and doubles the passive
leak on strict dendrite intervals. It records the chosen CSV row, projection
distance, all match/rejection counts, source metadata, and every electrical
assumption. This tests source-guided input placement; the clamp is not a
receptor model. For another experiment, use ``make_annotated_cell`` or compose
the imported selections with your own channels and inputs. The CLI also accepts
``--precision 32``. The full demo morphology showed about 0.05 mV drift in the
32-bit stimulated run. A 64-bit zero-input control still drifted by up to
0.00241 mV, failing a stricter 0.000001 mV equilibrium check. Thus finite output
is not an accuracy certificate; compare against a control and qualify numerical
accuracy for your experiment. Library helpers respect the caller's precision
environment and do not change it globally.

The active-model investigation subsequently reproduced an integer initialization
issue: ``V_init=-65 * u.mV`` created an integer state. The helper now uses
``-65.0 * u.mV``. Repeating the annotated 64-bit zero-input control after this
fix gave maximum rest error 7.86e-11 mV over 1 ms at dt=0.025 ms. The earlier
records above remain historical evidence; the corrected result is in
``docs/evidence/h01-active-rest-float64.json``. This check does not establish
long-duration or active-model numerical accuracy.

Annotation data come from ``proofread_104/segment_properties/info`` and
``proofread_104/synapse_locations.csv`` in the official release bucket.
Downloads validate pinned SHA-256 values after transport decompression and
stage each asset before replacement. Corrupt caches raise. Source verification
status is recorded separately from proofread morphology status.

Experimental active transfer (single-cell tool qualification pending)
--------------------------------------------------------------------

The active-model work adds Wilbers 2023 human sodium and mixed potassium
kinetics. These are borrowed human population models, not measurements of
the H01 donor cells. The default conductances are source-script starting
values; they are not a completed physiological fit.

.. code-block:: shell

   python -m braintrace.datasets.h01_active --archive ./data/h01/104_proofread_neurons_swc.zip --annotations ./data/h01 --active-radius-um 10 --current-na 0.5 --output ./active-run.json

The required radius explicitly chooses a neighborhood along the cable from
the largest-radius soma-labelled sample. This is an inferred active region,
not a verified soma or AIS boundary. The rest of the morphology is passive.
The command runs in 64-bit precision, uses compiled ``Cell.run``, and saves
source identity, assumptions, electrical parameters and the voltage trace.
Two transfers must be told apart. For this single-cell ``h01_active`` tool,
numerical and biological qualification are still in progress: the default
10 um maximum CV length is not spatially converged for the current
experimental transfer. The I-cell BrainCell-vs-NEURON transfer gate (SP2) is
not pending: it closed on 2026-09-07 on simulator identity (rise 1.2e-9 ms,
peak 5.4e-6 mV, width 1.9e-11 ms), with a remaining dt requirement at 0.19 nA
(1.864 ms at dt 0.000625, cap spent;
``docs/evidence/h01-i-transfer/sp2-close-decision.json``). Use ``--max-cv-length-um`` and ``--dt-ms`` for refinement;
do not treat a visible spike as validation.

Use ``--align-active-boundaries`` to split the mesh at active-region endpoints.
Keep this policy fixed during spatial refinement.
Use ``--pulse-count 5 --period-ms 25 --duration-ms 130`` for five pulses.
The default pulse width is 3 ms. Each pulse uses the specified current.
Repeated spikes alone do not establish agreement with human recordings.

Add ``--observe-channels`` with ``--align-active-boundaries`` to save local
gate states and sodium and potassium current densities.
These probes share an active compartment midpoint near the soma.
The output records that location separately from the original soma probe.
Current density is in uA/cm2, with positive current into the cell.

The :download:`causal model <../h01-causal-model.md>` defines measurement
datums and the nested path from channel gates to circuit behavior.
It includes a causal diagram and separates measured data from model assumptions.
The :download:`direct trace figure <../evidence/h01-causal-traces.svg>` shows
the observed simulation response to conductance changes.

Library callers can use ``braintrace.datasets.h01_active.make_active_cell``
or build the separate 10 by 10 um reference cylinder with
``braintrace.datasets.h01_reference.make_reference_cell``. The latter has
an independent numerical oracle test; it is not H01 anatomy. Neither entry
point implements an inhibitory neuron or a connected circuit yet.

The measured E/I pair: what its behaviour represents
---------------------------------------------------

``python -m examples.h01_ei_circuit --control ei --output .cache/h01/circuit-ei``
runs the two-cell circuit on measured anatomy: I ``5584343344`` to E
``4157825456`` through the released contact ``8105899``. Run
``--control disconnected`` with the same settings for the paired control.
Read the result with the following table in hand; every row is backed by a
record under ``docs/evidence``.

.. list-table::
   :header-rows: 1

   * - Element
     - Status
     - Meaning of the output
   * - Cell geometry and the contact endpoints
     - Measured (H01 release, checksum pinned)
     - Positions and cable lengths are the donor's; the contact is one released
       detection, not a manually verified synapse.
   * - Electrical regions
     - Inferred from sparse labels
     - The soma-to-contact path carries dendrite parameters; the map is a model
       hypothesis.
   * - Channel laws and densities
     - Borrowed (HL5BN1 for I, Allen 626170538 for E)
     - Under the command-only input (the recorded bias is a holding current the
       fit absorbed, ``h01-pv-bias-forensics.md``) the I finalist reproduces
       the spike loop and the trough at three inputs within the repeatability
       limits and predicted 9 of 11 rows on the 0.23 nA holdout, which still
       fails the approved acceptance contract on count and first peak
       (``h01-i-energetic-result.md``, ``h01-prediction-i.md``); its
       early burst and accommodation are unexplained. The E cell's rise rate is
       two to three times the human's and load-limited
       (``h01-e-energetic-stage-r-budgets.md``); the E cell passes neither the
       1 mV contract nor the usable tier as a whole, though individual rows
       pass (250 pA rate and several width rows,
       ``h01-usable-tier-e.json``). H01 has no recording to compare against.
   * - I output event
     - Diagnostic only
     - The I cell emits a contact event only with the sodium closing-time
       factor restored from 0.15 to 1.0 (``--restore-closing`` in the
       all-CV recorder). That factor is the sodium closing that gives the
       donor model its human-like loop (sodium inflow must end within the
       upstroke), so the default profile and the override cannot both be
       right; the H01 delivery question is about the initiation site, not the
       channel law.
   * - Synaptic conductance, delay, reversal
     - Borrowed settings
     - Delivery, its local sign, and a bounded onset delay are demonstrated;
       strength and kinetics are assumptions.
   * - Reciprocal E-to-I wiring
     - Candidate contact ``54906016`` unverified
     - Not in the default circuit; ``--connectivity illustrative`` uses a
       different, explicitly illustrative pair.

The one-page energetic map ``docs/evidence/h01-circuit-z-map.md`` places these
elements on the source-to-load path and names the parasitic elements the
campaigns did not partition. Training against this circuit is out of scope
until a cell passes the contract.

Two further pages state the circuit in the terms a user feels. The usable tier
(``docs/evidence/h01-usable-tier-i.md``, ``h01-usable-tier-e.md``) scores firing
rate, adaptation, spike width and afterhyperpolarisation from a retained per-cycle
table beside the 1 mV contract; neither cell passes it yet. The population status
page (``docs/evidence/h01-population-status.md``) lists, layer by layer, what the
104-cell network contains today: 55 of 104 cells with a type-matched donor across
four frozen donor profiles (Allen 541563728 L2, HL5BN1 PV, Allen 527952884 L4,
HL5MN1 L3; the two added donors' published-fit reproductions were rejected,
``docs/evidence/h01-donors/stage-*-decision.json``), three verified contacts,
assumed synaptic numbers with literature pins recorded, a four-cell construction
(one-step run), a measured 12-cell network (84,097 compartments) that ran 1 ms in
ei and disconnected arms with finite traces
(``docs/evidence/h01-population-build-12.json``), and a 40-cell build that FAILED
(``H01Archive.load`` ValueError on cell 5805562981; 7 of 104 largest components
fail to load; stages untested, runs stopped by the user;
``docs/evidence/h01-population-build-40.json``). The E cell gained an axonal initiation
site on 2026-09-07 (``docs/evidence/h01-e-usable/stage-a-decision.json``), which
moved its threshold and upstroke shape to the human's. It stays at the frozen,
unpromoted B3 profile: the gain split of its passive family closed as FAIL on
2026-09-08 (``docs/evidence/h01-e-gain-result.md``), leaving the low-drive count
(4 spikes at 200 pA against the donor's 1, 8 against 5 at 250 pA) to a
voltage- or use-dependent current the fit does not have.

Provenance and license
---------------------

Source: `H01 released data <https://h01-release.storage.googleapis.com/data.html>`_,
``20210601/proofread_104/skeletons/104_proofread_neurons_swc.zip``.
The archive's SHA-256 is
``3e0534df357ef2e92f6e0199962133cc9bc9733eb3a1fd6d1d315208ad63db47``.
Every loaded component includes its own source-byte SHA-256.

Credit Shapson-Coe et al., Science 384, eadk4858 (2024), the Lichtman laboratory
at Harvard University, and Connectomics at Google. H01 data are licensed under
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_. Retain attribution
and identify the coordinate and labeling transformations when redistributing
converted data; the package's Apache license does not replace H01's data license.

The original project's
`coordinate and annotation definitions <https://github.com/ashapsoncoe/h01/blob/db223b8d769d8ae046f0026f12b251103f1dbab0/get_separate_components_of_neurons.py>`_
and
`graph conversion <https://github.com/ashapsoncoe/h01/blob/db223b8d769d8ae046f0026f12b251103f1dbab0/common_functions_h01.py>`_
support this conversion. An independent comparison against H01's original
binary skeleton confirmed exact positions and radius agreement within SWC
rounding precision. See ``docs/evidence/h01-braincell-import-2026-09-04.md``
for the executed verification and its limits.
