"""Experimental staggered solver with implicit H01 calcium self states."""
import brainstate
import brainunit as u
import jax.numpy as jnp
from braincell import IndependentIntegration
from braincell.quad import register_integrator, ind_exp_euler_step
from .h01_pv_calcium import PVCalcium
from .h01_calcium_implicit import calcium_backward_euler
from .h01_dhs_scan import _voltage_step


def _get_node_ca_constants(node):
    cached = getattr(node, "_h01_ca_constants", None)
    if cached is not None:
        return cached
    factor = (node.gamma*(1.*u.mA/u.cm**2)/(2*u.faraday_constant*node.depth)).to_decimal(u.mM/u.ms)
    nernst = (u.gas_constant*node.temp/(node.valence*u.faraday_constant)).to_decimal(u.mV)
    decay = node.decay.to_decimal(u.ms)
    rest = node.rest.to_decimal(u.mM)
    Co = node.Co.to_decimal(u.mM)
    reversal = node.E.to_decimal(u.mV)
    cached = (factor, nernst, decay, rest, Co, reversal)
    try:
        object.__setattr__(node, "_h01_ca_constants", cached)
    except (AttributeError, TypeError):
        pass
    return cached


def _calcium_snapshot(node, voltage):
    """Capture old voltage and validate an ohmic calcium current sum."""
    zero = voltage*0.
    i0 = node.current(zero, include_external=True)
    i1 = node.current(zero+1.*u.mV, include_external=True)
    g = ((i0-i1)/(1.*u.mV)).to_decimal(u.mS/u.cm**2)
    actual = node.current(voltage, include_external=True).to_decimal(u.mA/u.cm**2)
    reversal = node.E.to_decimal(u.mV)
    predicted = (g*1e-3)*(reversal-voltage.to_decimal(u.mV))
    scale = jnp.maximum(jnp.abs(actual), 1e-12)
    valid = (jnp.abs(actual-predicted) <= 1e-9*scale) & (g >= 0)
    return voltage, jnp.where(valid, g, jnp.nan)


def _advance_calcium(node, voltage, conductance, dt):
    """Advance only concentration using the source constants and signed flux."""
    factor, nernst, decay, rest, Co, _ = _get_node_ca_constants(node)
    node.Ci.value = calcium_backward_euler(node.Ci.value.to_decimal(u.mM),
        voltage.to_decimal(u.mV), conductance, dt.to_decimal(u.ms),
        decay, factor, nernst, rest, Co)*u.mM


@register_integrator('h01_staggered_calcium_implicit', category='staggered',
    description='Experimental H01 scan voltage with backward-Euler Nernst calcium.')
def _implicit_step(target, *args):
    """Preserve family ordering while replacing H01 calcium self integration."""
    if target.ion_channel_update_order != 'family':
        raise ValueError('Implicit H01 calcium requires family ordering.')
    t, dt = brainstate.environ.get('t', 0.), brainstate.environ.get('dt')
    ions = target._family_ion_nodes()
    channels = target._family_channel_nodes()
    point_old = target._cv_to_point_unchecked(target.V.value)
    snapshots = [(node, *_calcium_snapshot(node, point_old))
                 for _, node in ions if isinstance(node, PVCalcium)]
    target.cache_ion_total_currents(target.V.value)
    _voltage_step(target, t, dt, *args)
    point_v = target._cv_to_point_unchecked(target.V.value)
    target._integrate_runtime_synapse_dynamics(point_v)
    paths = [path for path,node in ions if not isinstance(node, (PVCalcium, IndependentIntegration))]
    target._integrate_selected_ion_self_states(ions, paths, point_v,
        excluded_paths=[('V',), *[path for path,_ in channels]])
    for _, node in ions:
        if isinstance(node, IndependentIntegration):
            node.ind_update(point_v, recursive_child=False)
    for node, old_v, conductance in snapshots:
        _advance_calcium(node, old_v, conductance, dt)
    for path, node in channels:
        channel, arguments = target._channel_integration_target_and_args(path, node, point_v)
        if target._is_independent_channel(node):
            channel.ind_update(*arguments)
        else:
            ind_exp_euler_step(channel, *arguments)
