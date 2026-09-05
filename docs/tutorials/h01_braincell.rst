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
uniformly manually verified cell-compartment classification. This adapter does
not interpolate those sparse labels or turn them into region-specific channels.

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

H01 synapse connectivity, synaptic strength, ion-channel kinetics, learning
rules and missing external inputs are not supplied by this morphology adapter.
The output is a measured anatomical reference with caller-selected dynamics,
not an emulation of a complete human cortical circuit.

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
