"""BrainCell compatibility fixtures for the Example 21 replacement."""

from __future__ import annotations

import braincell
import brainstate
import braintrace
import brainunit as u
import braintools
import brainevent
import jax
import jax.numpy as jnp


N_FEATURES = 1
N_CELLS = 4
N_READOUT = 360
FINITE_DIFFERENCE_EPSILON = 1e-3
OBJECTIVE_VOLTAGE_OFFSET = 65.0
OBJECTIVE_VOLTAGE_SCALE = 20.0
SPIKE_DRIVE = 20.0


class CompatibilityHodgkinHuxley(braincell.SingleCompartment):
    """Construct the four-cell BrainCell 0.1.0 Hodgkin–Huxley fixture."""

    def __init__(self, size: int = N_CELLS):
        super().__init__(
            size,
            length=10.0 * u.um,
            radius=5.0 * u.um,
            C=1.0 * u.uF / u.cm**2,
            V_th=0.0 * u.mV,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            spk_fun=braintools.surrogate.ReluGrad(alpha=0.3, width=1.0),
            solver="ind_exp_euler",
        )
        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(
            INa=braincell.channel.Na_HH1952(
                size,
                g_max=120.0 * u.mS / u.cm**2,
                temp=309.15 * u.kelvin,
                temp_ref=309.15 * u.kelvin,
                V_sh=-45.0 * u.mV,
                q10=3.0,
            )
        )
        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(
            IK=braincell.channel.K_HH1952(
                size,
                g_max=10.0 * u.mS / u.cm**2,
                temp=309.15 * u.kelvin,
                temp_ref=309.15 * u.kelvin,
                V_sh=-45.0 * u.mV,
                q10=3.0,
            )
        )
        self.IL = braincell.channel.IL(
            size,
            g_max=0.03 * u.mS / u.cm**2,
            E=-54.387 * u.mV,
        )


def input_csr() -> brainevent.CSR:
    """Return the declared one-feature by four-cell CSR relation."""

    return brainevent.CSR(
        jnp.asarray([0.1, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        jnp.asarray([0, 4], dtype=jnp.int32),
        shape=(1, 4),
    )


def recurrent_csr() -> brainevent.CSR:
    """Return the four-cell recurrent CSR relation."""

    return brainevent.CSR(
        jnp.ones((N_CELLS,), dtype=jnp.float32),
        jnp.arange(N_CELLS, dtype=jnp.int32),
        jnp.arange(N_CELLS + 1, dtype=jnp.int32),
        shape=(N_CELLS, N_CELLS),
    )


def bounded_current_density(input_drive, recurrent_drive=0.0):
    """Convert bounded dimensionless drives to current density."""

    return (
        0.02 * jnp.tanh(input_drive / 20.0) * u.mA / u.cm**2
        + 0.01 * jnp.tanh(recurrent_drive / 20.0) * u.mA / u.cm**2
    )


class PPPropRelationFixture(brainstate.nn.Module):
    """Expose input and recurrent sparse weights to the PP-Prop compiler."""

    def __init__(self, input_weight=0.1):
        super().__init__()
        self.hidden = brainstate.HiddenState(jnp.zeros((1, N_CELLS)))
        self.cell = CompatibilityHodgkinHuxley()
        self.cell.init_state()
        self.cell.reset_state()
        self.input_weight = brainstate.ParamState(
            jnp.asarray([input_weight, 0.0, 0.0, 0.0])
        )
        self.recurrent_weight = brainstate.ParamState(jnp.ones((N_CELLS,)))
        self._input_csr = input_csr()
        self._recurrent_csr = recurrent_csr()

    def update(self, x):
        hidden = braintrace.sparse_matmul(
            x,
            self.input_weight.value,
            sparse_mat=self._input_csr,
        )
        recurrent = braintrace.sparse_matmul(
            self.hidden.value,
            self.recurrent_weight.value,
            sparse_mat=self._recurrent_csr,
        )
        drive = hidden + recurrent
        with brainstate.environ.context(dt=0.1 * u.ms):
            self.cell.update(bounded_current_density(drive))
        self.hidden.value = self.cell.V.value.to_decimal(u.mV)
        return self.hidden.value


def pp_prop_relation_fixture():
    """Compile and return the two sparse hidden-state relations."""

    model = PPPropRelationFixture()
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((1, N_FEATURES), dtype=jnp.float32),
        batch_size=1,
        decay_or_rank=0.95,
        vjp_method="single-step",
    )
    return learner.graph.hidden_param_op_relations


def _pp_prop_gradient(input_weight):
    model = PPPropRelationFixture(input_weight)
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((1, N_FEATURES), dtype=jnp.float32),
        batch_size=1,
        decay_or_rank=0.95,
        vjp_method="single-step",
    )

    def step_loss(x):
        voltage = learner(x)
        return jnp.mean(
            jnp.tanh((voltage + OBJECTIVE_VOLTAGE_OFFSET) / OBJECTIVE_VOLTAGE_SCALE)
        )

    gradients = learner.etrace_grad(
        jnp.ones((1, 1, N_FEATURES), dtype=jnp.float32),
        step_fn=step_loss,
        reduction="sum",
    )
    return gradients[("input_weight",)][0]


def _pp_prop_objective(input_weight):
    return _braincell_objective(input_weight)["objective"]


def _csr_input_drive(input_weight):
    weights = jnp.asarray([input_weight, 0.0, 0.0, 0.0])
    return braintrace.sparse_matmul(
        jnp.ones((1, N_FEATURES)), weights, sparse_mat=input_csr()
    ).reshape((N_CELLS,))


def advance_one_step(cell: CompatibilityHodgkinHuxley, current):
    """Advance a fixture cell by one compiled 0.1 ms interval."""

    with brainstate.environ.context(dt=0.1 * u.ms):
        cell.update(current)
    return cell.V.value, cell.spike.value


def compiled_one_step(cell: CompatibilityHodgkinHuxley):
    """Return a JIT-compiled one-step driver for a fixture cell."""

    return brainstate.transform.jit(advance_one_step, static_argnums=0)(
        cell, bounded_current_density(0.0)
    )


def finite_difference_fixture() -> dict[str, float]:
    """Compare one PP-Prop BrainCell step with a central difference."""

    weight = jnp.asarray(0.1, dtype=jnp.float32)
    epsilon = FINITE_DIFFERENCE_EPSILON

    relations = pp_prop_relation_fixture()
    direct = float(_pp_prop_gradient(weight))
    plus = _braincell_objective(weight + epsilon)
    minus = _braincell_objective(weight - epsilon)
    centered = float((plus["objective"] - minus["objective"]) / (2.0 * epsilon))
    tolerance = 1e-5 + 1e-2 * max(abs(direct), abs(centered))
    return {
        "pp_prop": direct,
        "finite_difference": centered,
        "absolute_error": abs(direct - centered),
        "tolerance": tolerance,
        "relations": float(len(relations)),
        "finite_voltage": float(
            bool(jnp.isfinite(plus["voltage"]) and jnp.isfinite(minus["voltage"]))
        ),
        "finite_gates": float(plus["finite_gates"] and minus["finite_gates"]),
        "zero_spikes": float(plus["zero_spikes"] and minus["zero_spikes"]),
        "reset_isolated": float(plus["reset_isolated"] and minus["reset_isolated"]),
    }


def _braincell_objective(input_weight):
    cell = CompatibilityHodgkinHuxley()
    cell.init_state()
    cell.reset_state()
    reset_copy = CompatibilityHodgkinHuxley()
    reset_copy.init_state()
    reset_copy.reset_state()
    reset_voltage = reset_copy.V.value.to_decimal(u.mV).copy()
    reset_gates = tuple(
        gate.value.copy()
        for gate in (reset_copy.na.INa.p, reset_copy.na.INa.q, reset_copy.k.IK.p)
    )
    current = bounded_current_density(_csr_input_drive(input_weight))
    voltage, spikes = advance_one_step(cell, current)
    objective = jnp.mean(
        jnp.tanh((voltage.to_decimal(u.mV) + OBJECTIVE_VOLTAGE_OFFSET) / OBJECTIVE_VOLTAGE_SCALE)
    )
    finite_gates = all(
        jnp.all(jnp.isfinite(gate))
        for gate in (cell.na.INa.p.value, cell.na.INa.q.value, cell.k.IK.p.value)
    )
    reset_isolated = all(
        jnp.allclose(gate.value, initial)
        for gate, initial in zip(
            (reset_copy.na.INa.p, reset_copy.na.INa.q, reset_copy.k.IK.p),
            reset_gates,
        )
    ) and jnp.allclose(reset_copy.V.value.to_decimal(u.mV), reset_voltage)
    return {
        "objective": objective,
        "voltage": jnp.mean(voltage.to_decimal(u.mV)),
        "finite_gates": bool(finite_gates),
        "zero_spikes": bool(jnp.all(spikes == 0)),
        "reset_isolated": reset_isolated,
    }


def spike_path_fixture() -> dict[str, bool]:
    """Check deterministic threshold crossing and finite surrogate activity."""

    cell = CompatibilityHodgkinHuxley()
    cell.init_state()
    cell.V.value = jnp.full((N_CELLS,), -0.001) * u.mV
    def spike_objective(input_drive):
        trial = CompatibilityHodgkinHuxley()
        trial.init_state()
        trial.reset_state()
        trial.V.value = jnp.full((N_CELLS,), -0.001) * u.mV
        _, spikes = advance_one_step(trial, bounded_current_density(input_drive))
        return jnp.sum(spikes)

    voltage, spikes = advance_one_step(cell, bounded_current_density(SPIKE_DRIVE))
    spike_gradient = jax.grad(spike_objective)(jnp.asarray(SPIKE_DRIVE))
    return {
        "threshold_crossed": bool(jnp.any(voltage >= 0.0 * u.mV)),
        "finite_voltage": bool(jnp.all(jnp.isfinite(voltage.mantissa))),
        "finite_spikes": bool(jnp.all(jnp.isfinite(spikes))),
        "finite_gradient": bool(jnp.all(jnp.isfinite(spike_gradient))),
        "nonzero_gradient": bool(jnp.any(spike_gradient != 0.0)),
    }


def direct_readout_gradient_fixture() -> dict[str, bool]:
    """Check finite direct gradients for all 360 voltage-readout values."""

    features = jnp.asarray([[0.1, -0.2, 0.3, -0.4]])
    weight = jnp.ones((N_CELLS, N_READOUT)) * 0.1
    bias = jnp.zeros((N_READOUT,))

    def objective(readout_weight, readout_bias):
        return jnp.sum(jnp.tanh(features @ readout_weight + readout_bias))

    grad_weight, grad_bias = jax.grad(objective, argnums=(0, 1))(weight, bias)
    return {
        "shape": grad_bias.shape == (N_READOUT,),
        "finite": bool(
            jnp.all(jnp.isfinite(grad_weight))
            and jnp.all(jnp.isfinite(grad_bias))
        ),
        "height_nonzero": bool(jnp.any(grad_weight[:, :30] != 0.0)),
        "width_nonzero": bool(jnp.any(grad_weight[:, 30:60] != 0.0)),
        "color_nonzero": bool(jnp.any(grad_weight[:, 60:] != 0.0)),
    }


def main() -> None:
    """Run the local compatibility checks."""

    print(finite_difference_fixture())
    print(spike_path_fixture())


if __name__ == "__main__":
    main()
