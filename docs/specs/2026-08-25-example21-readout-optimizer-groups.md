# Example 21 inferred readout optimizer groups

## Requirement

When a PP-Prop episode trainer infers direct readout `ParamState` values from
the compiled model, those values SHALL be included in the Adam group state
before the first episode update.

## Scenario

- **WHEN** a trainer is constructed with compiled biological parameters and a
  model exposes `readout_weight` and `readout_bias`
- **THEN** both readout values are present in the trainer parameter map
- **AND** both have independent Adam moment state
- **AND** a direct readout gradient updates both values on the first episode

