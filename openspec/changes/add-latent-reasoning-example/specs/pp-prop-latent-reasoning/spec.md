## Purpose

Demonstrates that a recurrent spiking network trained by pp_prop can acquire a
rule from demonstrations at inference time, hold it in a contextual memory
written without changing parameters, compute with it across repeated latent
iterations, and reports what that latent space looks like and whether iterating
it changes the outcome.

## ADDED Requirements

### Requirement: Episode structure

An episode SHALL consist of a demonstration phase, a query phase, and a latent
phase, in that order, over a single contiguous time axis. The demonstration
phase SHALL present `K` demonstration pairs drawn from one rule; the query phase
SHALL present one held-out query input; the latent phase SHALL present no
external input for `R` steps. Supervision SHALL be applied only at the end of
the latent phase.

#### Scenario: Phases occupy disjoint, ordered spans

- **WHEN** an episode is generated with `K` demonstrations and `R` latent steps
- **THEN** every time step belongs to exactly one of the three phases, the
  phases appear in demonstration-query-latent order, and the latent span carries
  zero external input

#### Scenario: Supervision is terminal

- **WHEN** the loss is computed for an episode
- **THEN** it reads only the final latent step's output, and no intermediate
  latent state is decoded into a target

#### Scenario: Zero latent steps is legal

- **WHEN** an episode is generated with `R = 0`
- **THEN** the episode is well-formed and the output is read directly from the
  state produced by the query phase

### Requirement: Fresh per-episode rule

Each episode SHALL draw a fresh bijection over `C` symbols, independently of
every other episode. Demonstration pairs SHALL be `(x, rule(x))` under that
episode's bijection. The query target SHALL be `rule(query)` under the same
bijection. No rule identifier SHALL be supplied as an input at any point.

#### Scenario: Oracle agreement

- **WHEN** an episode's demonstration pairs and query target are generated
- **THEN** every pair and the target are consistent with a single bijection over
  the symbol set, verified independently of the generator that produced them

#### Scenario: Rules vary across episodes

- **WHEN** many episodes are generated from one seed sequence
- **THEN** more than one distinct bijection appears, so a model cannot succeed by
  memorizing a fixed mapping

#### Scenario: No rule leakage through inputs

- **WHEN** the model input for an episode is inspected
- **THEN** it contains only encoded symbols and phase information, and contains
  no encoding of the bijection, the episode index, or the target

### Requirement: Contextual memory written with parameters frozen

During the demonstration phase the system SHALL accumulate a contextual memory
from demonstration-driven activity. The step function SHALL NOT write to any
trainable parameter. Parameter updates SHALL occur only through the learning
algorithm's own update applied to a training episode as a whole; the parameters
that shape memory-writing are trained by that update like any others.

#### Scenario: The step function performs no parameter write

- **WHEN** a single time step of any phase is executed
- **THEN** the step writes only to state, and every trainable parameter is
  bitwise identical before and after, while the contextual memory has changed
  during the demonstration phase

#### Scenario: Frozen evaluation changes no parameters across a whole episode

- **WHEN** a complete episode is run against a frozen model during the
  intervention grid
- **THEN** every trainable parameter is bitwise identical before and after the
  entire episode, and the contextual memory reflects that episode's
  demonstrations

#### Scenario: Memory content depends on demonstrations

- **WHEN** two episodes with identical query inputs but different demonstration
  pairs are run
- **THEN** the resulting contextual memories differ

#### Scenario: Memory capacity is bounded and reported

- **WHEN** the contextual memory is configured for a given number of
  demonstrations
- **THEN** its storage grows proportionally to the number of demonstrations and
  the state width, not to the square of the state width, and the configured
  capacity is reported

### Requirement: Context support intervention

The system SHALL support generating held-out evaluation episodes in matched
*supported* and *short* conditions, where the query input is byte-identical
across conditions and the conditions differ only in whether the queried symbol's
binding appears among the demonstrations.

#### Scenario: Queries are byte-identical across conditions

- **WHEN** a matched supported/short pair of episodes is generated
- **THEN** the query-phase inputs and the targets are byte-identical, and only
  the demonstration-phase content differs

#### Scenario: Short condition omits the queried binding

- **WHEN** an episode is generated in the short condition
- **THEN** no demonstration pair contains the queried symbol on either side of
  the mapping

#### Scenario: Supported condition includes the queried binding

- **WHEN** an episode is generated in the supported condition
- **THEN** exactly one demonstration pair binds the queried symbol to its target

### Requirement: Latent iteration is a configurable depth

The number of latent iterations `R` SHALL be configurable at run time and SHALL
be reported with every result. Accuracy SHALL be reported for each configured
`R`, including `R = 0`.

#### Scenario: Depth sweep is reported

- **WHEN** the example is run with a sweep over latent depths
- **THEN** the report contains one accuracy entry per depth, each labeled with
  its depth, and `R = 0` is present as the no-iteration control

#### Scenario: Depth changes nothing else

- **WHEN** two runs differ only in `R`
- **THEN** the demonstration and query phases, the rule draws, and the parameter
  initialization are identical between them

### Requirement: Binding-count sweep

The system SHALL support varying the number of simultaneous bindings presented
in the demonstration phase across at least the range two through eight, and
SHALL report accuracy per binding count.

#### Scenario: Binding count is swept and reported

- **WHEN** the example is run with a binding-count sweep
- **THEN** the report contains one accuracy entry per binding count in the
  configured range

#### Scenario: Binding count exceeding memory capacity is rejected or reported

- **WHEN** a binding count larger than the configured memory capacity is
  requested
- **THEN** the run either fails with a clear error naming both numbers, or
  proceeds and reports the overflow explicitly; it never silently discards
  bindings

### Requirement: Shuffled-memory control

The system SHALL provide a control arm in which the contextual memory's stored
associations are permuted before the query and latent phases, leaving the memory
present but its content mismatched to the demonstrations.

#### Scenario: Shuffled control is reported alongside the intact arm

- **WHEN** the example is run
- **THEN** the report contains accuracy for both the intact and shuffled-memory
  arms under otherwise identical conditions

#### Scenario: Shuffling preserves memory shape and magnitude

- **WHEN** the memory is shuffled
- **THEN** its dimensions are unchanged and its overall magnitude is preserved,
  so the arms differ in content rather than in the presence or scale of memory

### Requirement: Latent geometry report

The system SHALL report, over held-out episodes, four measurements of the latent
phase: the effective dimensionality of the latent state at each iteration, the
step-to-step change in the latent state across iterations, the linear
decodability of the query answer from the latent state at each iteration, and
the linear decodability of the query answer from the contextual memory alone.

#### Scenario: Per-iteration measurements are emitted

- **WHEN** a run completes with `R` latent iterations
- **THEN** the report contains effective dimensionality, step-to-step change, and
  answer decodability for each iteration index from zero through `R`

#### Scenario: Probes are fit and scored on disjoint episodes

- **WHEN** a decodability probe is fit
- **THEN** it is fit on one set of episodes and scored on a disjoint set, and the
  report states both counts

#### Scenario: Memory-only decodability is reported for comparison

- **WHEN** the report is produced
- **THEN** decodability from the contextual memory alone appears beside
  decodability from the final latent state, so the reader can see whether
  iteration added anything

#### Scenario: Answer and rule decodability are distinguished

- **WHEN** decodability is reported
- **THEN** decoding the current query's answer is reported separately from
  decoding the full rule, and the two are not conflated in any summary line

#### Scenario: A null separation result is reported, not suppressed

- **WHEN** decodability from the contextual memory alone matches or exceeds
  decodability from the final latent state
- **THEN** the report states plainly that the two-state separation did not add
  decodable information at this scale, rather than omitting the comparison

### Requirement: Reproducibility and smoke path

A run SHALL be reproducible from its reported configuration and seed, and the
example SHALL provide a reduced smoke configuration.

#### Scenario: Same seed reproduces reported metrics

- **WHEN** the example is run twice with the same configuration and seed on the
  same device
- **THEN** the reported accuracies and geometry measurements agree to the
  reported numerical tolerance

#### Scenario: Smoke path completes quickly

- **WHEN** the example is run in smoke mode
- **THEN** it exercises every phase, arm, and reported measurement at reduced
  size and completes fast enough to serve as an iteration check

### Requirement: Claim boundary

The example SHALL state that it instantiates only the published system-level
interface of the source work and SHALL NOT claim reproduction of that system,
report any benchmark score attributed to it, or make any inference-cost claim.
The example SHALL NOT assert any property of the learning algorithm's gradient
estimate.

#### Scenario: Non-claims are stated in the example and its spec

- **WHEN** the example's documentation and report are read
- **THEN** both state that internal update rules and dimensions of the source
  system are unpublished, that this is an instantiation of the interface only,
  and that no benchmark or cost claim is made

#### Scenario: No gradient-correctness assertion is present

- **WHEN** the example's tests are inspected
- **THEN** none of them asserts agreement, or a bounded deviation, between the
  online gradient estimate and a backpropagation-through-time oracle

### Requirement: Input validation

Malformed configuration and malformed episode data SHALL raise a clear error
naming the offending quantity rather than producing a misleading result.

#### Scenario: Contradictory sizes are rejected

- **WHEN** a configuration requests more distinct demonstration bindings than
  there are symbols, a negative latent depth, or a probe split leaving either
  side empty
- **THEN** the run raises an error naming the offending quantity and does not
  produce a report

#### Scenario: Misaligned analysis inputs are rejected

- **WHEN** latent states, labels, or memory factors with mismatched leading
  dimensions are passed to the geometry report
- **THEN** an error is raised naming the mismatched shapes
