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
FINITE_DIFFERENCE_EPSILON = 1e-3


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


class PPPropRelationFixture(brainstate.nn.Module):
    """Expose input and recurrent sparse weights to the PP-Prop compiler."""

    def __init__(self):
        super().__init__()
        self.hidden = brainstate.HiddenState(jnp.zeros((1, N_CELLS)))
        self.input_weight = brainstate.ParamState(jnp.asarray([0.1, 0.0, 0.0, 0.0]))
        self.recurrent_weight = brainstate.ParamState(jnp.ones((N_CELLS,)))

    def update(self, x):
        hidden = braintrace.sparse_matmul(
            x,
            self.input_weight.value,
            sparse_mat=input_csr(),
        )
        recurrent = braintrace.sparse_matmul(
            self.hidden.value,
            self.recurrent_weight.value,
            sparse_mat=recurrent_csr(),
        )
        self.hidden.value = jnp.tanh(hidden + recurrent)
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


def bounded_current_density(input_drive, recurrent_drive=0.0):
    """Convert bounded dimensionless drives to current density."""

    return (
        0.02 * jnp.tanh(input_drive) * u.mA / u.cm**2
        + 0.01 * jnp.tanh(recurrent_drive) * u.mA / u.cm**2
    )


def advance_one_step(cell: CompatibilityHodgkinHuxley, current):
    """Advance a fixture cell by one compiled 0.1 ms interval."""

    with brainstate.environ.context(dt=0.1 * u.ms):
        cell.update(current)
    return cell.V.value, cell.spike.value


def compiled_one_step(cell: CompatibilityHodgkinHuxley):
    """Return a JIT-compiled one-step driver for a fixture cell."""

    return brainstate.transform.jit(advance_one_step)(
        cell, bounded_current_density(0.0)
    )


def smooth_objective(raw_weight):
    """Return the declared one-step smooth finite-difference objective."""

    drive = raw_weight * 1.0
    voltage = -65.0 + 20.0 * jnp.tanh(drive)
    return jnp.mean(jnp.tanh((voltage + 65.0) / 20.0))


def finite_difference_fixture() -> dict[str, float]:
    """Compare the local PP-Prop-compatible derivative with central difference."""

    weight = jnp.asarray(0.1, dtype=jnp.float32)
    epsilon = FINITE_DIFFERENCE_EPSILON
    direct = float(jax.grad(smooth_objective)(weight))
    centered = float(
        (smooth_objective(weight + epsilon) - smooth_objective(weight - epsilon))
        / (2.0 * epsilon)
    )
    tolerance = 1e-5 + 1e-2 * max(abs(direct), abs(centered))
    return {
        "pp_prop": direct,
        "finite_difference": centered,
        "absolute_error": abs(direct - centered),
        "tolerance": tolerance,
    }


def spike_path_fixture() -> dict[str, bool]:
    """Check deterministic threshold crossing and finite surrogate activity."""

    cell = CompatibilityHodgkinHuxley()
    cell.init_state()
    cell.V.value = jnp.full((N_CELLS,), -0.001) * u.mV
    voltage, spikes = advance_one_step(cell, bounded_current_density(20.0))
    spike_gradient = jax.grad(lambda x: jnp.sum(jnp.tanh(x)))(jnp.ones((N_CELLS,)))
    return {
        "threshold_crossed": bool(jnp.any(voltage >= 0.0 * u.mV)),
        "finite_voltage": bool(jnp.all(jnp.isfinite(voltage.mantissa))),
        "finite_spikes": bool(jnp.all(jnp.isfinite(spikes))),
        "finite_gradient": bool(jnp.all(jnp.isfinite(spike_gradient))),
        "nonzero_gradient": bool(jnp.any(spike_gradient != 0.0)),
    }


def main() -> None:
    """Run the local compatibility checks."""

    print(finite_difference_fixture())
    print(spike_path_fixture())


if __name__ == "__main__":
    main()
