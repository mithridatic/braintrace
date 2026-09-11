"""Numerical contracts for H01 sparse Muon."""
from pathlib import Path
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from .example21_arc_adapter import Example21ArcAdapter
from .h01_muon import EdgeLayout, H01Muon, POLICY, contact_slots
from .h01_remap import remap_optimizer_group


def fixture(edges=4, cells=2):
    coords = np.array([(i//(cells*cells), (i//cells)%cells, i%cells)
                       for i in range(edges)], dtype=int).reshape(-1, 3)
    recurrent = EdgeLayout((max(1, (edges+cells*cells-1)//(cells*cells)), cells, cells), coords)
    inputs = EdgeLayout((1, cells, cells), np.array([(0,i,j) for i in range(cells)
                                                    for j in range(cells)]))
    params = dict(input=jnp.linspace(-.3,.7,cells*cells), recurrent=jnp.linspace(.1,.8,edges),
                  readout_weight=jnp.ones((cells,3))*.2, readout_bias=jnp.zeros(3))
    return H01Muon(dict(input=inputs,recurrent=recurrent),params), params


def test_sparse_connection_update_is_not_adamw():
    adapter, params = fixture()
    module = Example21ArcAdapter(Path('.'))._model()
    trainer = module.PPPropEpisodeTrainer(None, params, optimizer_adapter=adapter)
    grads = jax.tree.map(lambda x: jnp.arange(x.size,dtype=x.dtype).reshape(x.shape)*.3+.1,params)
    updated, _ = trainer.optimizer_adapter.update(params,grads,trainer.muon_groups)
    adam = optax.adamw(.001,weight_decay=.1,nesterov=True)
    delta, _ = adam.update(grads['input'],adam.init(params['input']),params['input'])
    assert not np.allclose(updated['input'], params['input']+delta,rtol=1e-5,atol=1e-7)


@pytest.mark.parametrize('edges,cells', [(0,2),(1,1),(4,2),(36,12),(240,12)])
def test_multistep_dense_optax_oracle(edges,cells):
    adapter, params = fixture(edges,cells)
    state = adapter.init(params)
    references, transforms, refstate = {}, {}, {}
    for name,p in params.items():
        layout = adapter.layouts.get(name)
        references[name] = layout.scatter(p) if layout else p
        transforms[name] = optax.contrib.muon(adapter.rates[name],weight_decay=.1,
            adam_learning_rate=adapter.rates[name],adam_weight_decay=.1,
            ns_coeffs=tuple(POLICY['ns_coeffs']),
            muon_weight_dimension_numbers=optax.contrib.MuonDimensionNumbers(1,2) if layout else None)
        refstate[name] = transforms[name].init(references[name])
    def step(carry, factor):
        p,s,r,rs = carry
        grads = jax.tree.map(lambda x: jnp.sin(jnp.arange(x.size).reshape(x.shape)+factor),p)
        new, ns = adapter.update(p,grads,s)
        nr,nrs = {},{}
        for name in p:
            layout = adapter.layouts.get(name)
            dense = layout.scatter(grads[name]) if layout else grads[name]
            delta,nrs[name] = transforms[name].update(dense,rs[name],r[name])
            result = r[name]+delta
            nr[name] = layout.scatter(layout.gather(result)) if layout else result
        errors = jnp.stack([jnp.max(jnp.abs(new[n]-(adapter.layouts[n].gather(nr[n])
            if n in adapter.layouts else nr[n])),initial=0.) for n in p])
        return (new,ns,nr,nrs),errors
    (_,final,_,_),errors = brainstate.transform.scan(step,(params,state,references,refstate),jnp.arange(1.,4.))
    np.testing.assert_allclose(errors,0,atol=2e-7)
    for name in ('input','recurrent','readout_weight'):
        assert final[name].mu.shape == params[name].shape
        assert int(final[name].count)==3
        assert np.isfinite(np.asarray(final[name].mu)).all()


def test_zero_gradient_decay_and_empty_edges():
    adapter,p = fixture(0,2)
    result,state = adapter.update(p,jax.tree.map(jnp.zeros_like,p),adapter.init(p))
    for name in p:
        np.testing.assert_allclose(result[name],p[name]*(1-.1*adapter.rates[name]),atol=1e-7)
    assert state['recurrent'].mu.size==0


def test_reordering_edges_does_not_change_updates():
    adapter,p = fixture(4,2)
    permutation = np.array([3,1,0,2])
    layouts = {n: EdgeLayout(l.shape,l.coordinates[permutation]) for n,l in adapter.layouts.items()}
    shuffled = {n: x[permutation] if n in layouts else x for n,x in p.items()}
    other = H01Muon(layouts,shuffled)
    g = jax.tree.map(lambda x: x*x+.13,p)
    gs = {n: x[permutation] if n in layouts else x for n,x in g.items()}
    a,_ = adapter.update(p,g,adapter.init(p))
    b,_ = other.update(shuffled,gs,other.init(shuffled))
    for name in layouts:
        np.testing.assert_allclose(a[name][permutation],b[name],atol=1e-7)


def test_parallel_contacts_have_separate_momentum():
    adapter,p = fixture(240,12)
    g = jax.tree.map(lambda x: jnp.arange(x.size,dtype=x.dtype).reshape(x.shape)+1,p)
    _,state = adapter.update(p,g,adapter.init(p))
    assert state['recurrent'].mu[0] != state['recurrent'].mu[144]
    np.testing.assert_allclose(state['recurrent'].mu,g['recurrent']*.05,rtol=1e-6)


def test_contact_slots_preserve_survivors_and_fill_holes():
    contacts = [('c','a','b'),('a','a','b'),('b','a','b'),('x','b','a')]
    assert contact_slots(contacts)==dict(a=0,b=1,c=2,x=0)
    slots=contact_slots([contacts[0],contacts[2],('new','a','b')],dict(a=0,b=1,c=2))
    assert slots==dict(c=2,b=1,new=0)
    for inherited in ({'b':-1},{'b':True},{'b':0,'c':0}):
        with pytest.raises(ValueError):
            contact_slots(contacts,inherited)
    with pytest.raises(ValueError):
        contact_slots(contacts+[contacts[0]])


def test_survivor_momentum_transport_and_next_update():
    adapter,p=fixture()
    _,s=adapter.update(p,p,adapter.init(p))
    remapped=remap_optimizer_group(s['recurrent'],adapter.init(p)['recurrent'],[2,-1,0,3],(4,),(4,))
    np.testing.assert_array_equal(remapped.mu,[s['recurrent'].mu[2],0,s['recurrent'].mu[0],s['recurrent'].mu[3]])
    assert int(remapped.count)==1
    restored=jax.tree.map(lambda x:jnp.asarray(np.array(x)),s)
    a,sa=adapter.update(p,p,s)
    b,sb=adapter.update(p,p,restored)
    for x,y in zip(jax.tree.leaves((a,sa)),jax.tree.leaves((b,sb))):
        np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('shape,coords', [((1,0,2),[[0,0,0]]),((1,2,2),[[0,0,0],[0,0,0]]),
    ((1,2,2),[[0,2,0]]),((1,2,2),[[0.,0.,0.]]),((1,2,2),[[0,-1,0]]),((1,2),[[0,0,0]])])
def test_invalid_layout(shape,coords):
    with pytest.raises(ValueError): EdgeLayout(shape,coords)


def test_invalid_shapes_groups_and_workspace():
    adapter,p=fixture()
    for cap in (0,True,1):
        with pytest.raises(ValueError): H01Muon(adapter.layouts,p,workspace_limit_bytes=cap)
    with pytest.raises(ValueError): H01Muon({},p)
    bad=dict(p,input=jnp.zeros(3))
    with pytest.raises(ValueError): H01Muon(adapter.layouts,bad)
    with pytest.raises(ValueError): H01Muon(adapter.layouts,dict(p,readout_weight=jnp.zeros(3)))
    with pytest.raises(ValueError): adapter.layouts['input'].scatter(jnp.zeros(3))
    state=adapter.init(p)
    with pytest.raises(ValueError): adapter.update(p,{},state)
    with pytest.raises(ValueError): adapter.update(p,bad,state)
    state['input']=state['input']._replace(mu=jnp.zeros(3))
    with pytest.raises(ValueError): adapter.update(p,p,state)
    meta=adapter.metadata()
    assert meta['policy']['weight_decay']==.1
    meta['policy']['learning_rates']['input']=42
    assert adapter.metadata()['policy']['learning_rates']['input']==.001
    assert adapter.report['recurrent']['algorithm']=='masked_muon'


def test_physical_delivery_order_and_inherited_parallel_slots():
    from types import SimpleNamespace as Obj
    from .h01_muon import model_optimizer
    _,p=fixture(2,2)
    blocks=[Obj(source=Obj(synapse=key,pre_population='cell_a',post_population='cell_b'))
            for key in ('syn_c','syn_a')]
    model=Obj(neuron_count=2,source_ids=('a','b'),
        stepper=Obj(network=Obj(populations={'cell_a':None,'cell_b':None}),
                    setup=Obj(delivery_blocks=blocks)),
        input_csr=Obj(indices=np.array([0,1,0,1]),indptr=np.array([0,2,*([4]*440)])))
    adapter=model_optimizer(model,p,inherited_slots={'syn_c':3,'syn_a':1})
    np.testing.assert_array_equal(adapter.layouts['recurrent'].coordinates,[[3,0,1],[1,0,1]])
    assert adapter.layouts['recurrent'].shape==(4,2,2)
    assert adapter.layouts['input'].shape==(1,441,2)
    assert adapter.metadata()['slots']=={'syn_c':3,'syn_a':1}
    blocks.append(Obj(source=Obj(synapse='syn_new',pre_population='cell_a',post_population='cell_b')))
    child=model_optimizer(model,dict(p,recurrent=jnp.ones(3)),inherited_slots=adapter.slots)
    assert child.slots=={'syn_c':3,'syn_a':1,'syn_new':0}
    blocks.pop(1)
    pruned=model_optimizer(model,p,inherited_slots=child.slots)
    assert pruned.slots=={'syn_c':3,'syn_new':0}


def test_adapter_maps_compiled_paths_and_writes_model_parameters():
    from types import SimpleNamespace as Obj
    adapter,p=fixture()
    module=Example21ArcAdapter(Path('.'))._model()
    paths=dict(input='input_weight',recurrent='recurrent_weight',
               readout_weight='readout_weight',readout_bias='readout_bias')
    states={(path,):brainstate.ParamState(jnp.zeros_like(p[name])) for name,path in paths.items()}
    trainer=module.PPPropEpisodeTrainer(Obj(param_states=states),p,optimizer_adapter=adapter)
    compiled={(path,):p[name] for name,path in paths.items()}
    mapped=trainer._group_gradients(compiled)
    assert mapped.keys()==p.keys()
    trainer._sync_compiled_parameters()
    for name,path in paths.items():
        np.testing.assert_array_equal(states[(path,)].value,p[name])
    assert trainer.adam_groups=={}
