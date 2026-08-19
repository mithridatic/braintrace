Online Learning Routines
========================

Take a model from definition to an executable online-learning workflow. These
chapters use compact tasks to make state handling, trace updates, and training
behavior inspectable.

.. note::

   Complete :doc:`../quickstart/quickstart` and
   :doc:`../quickstart/concepts` first if you have not yet compiled a
   BrainTrace model.

Choose a workflow
-----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: RNN Online Learning
      :link: rnn_online_learning
      :link-type: doc

      Train a GRU on the copying task with D-RTRL, then compare the online
      workflow with a BPTT baseline.

      **Best for:** continuous hidden states and sequence-memory tasks.

   .. grid-item-card:: SNN Online Learning
      :link: snn_online_learning
      :link-type: doc

      Build a recurrent LIF network, train it with pp-prop, and inspect how
      factorized traces differ from D-RTRL.

      **Best for:** spike-based dynamics, surrogate gradients, and physical
      units.

Two update schedules
--------------------

Both workflows above accumulate. The learning rule is a sum of per-step terms,

.. math::

   \nabla_{\theta} \mathcal{L} = \sum_{t' \in \mathcal{T}}
   \frac{\partial \mathcal{L}^{t'}}{\partial \mathbf{h}^{t'}}
   \circ \boldsymbol{\epsilon}^{t'},

and every term is complete at its own timestep -- nothing in a term refers to a
later step. That is what an eligibility trace buys, and it leaves a choice about
*when* to spend it.

:meth:`~braintrace.SequenceDriverMixin.etrace_grad` adds the terms up and takes
one optimizer step once the sequence ends. The gradient is computed online; the
update is not. This is the right default when one sequence is one training
example.

:meth:`~braintrace.SequenceDriverMixin.etrace_online` hands each term to the
optimizer as it is produced, so step ``t + 1`` runs under parameters that step
``t`` already moved:

.. code-block:: python

   opt = braintools.optim.Adam(1e-3)
   opt.register_trainable_weights(learner.param_states)

   learner.etrace_online(
       inputs, targets,
       step_fn=step_loss,
       optimizer=opt,
       transform=lambda g: brainstate.nn.clip_grad_norm(g, 1.0),
   )

Reach for it when the weights should change *during* the sequence -- test-time
adaptation, continual streams, or any setting where waiting for the sequence to
end means waiting too long.

Two differences are easy to miss:

- ``mask`` gates the **update** as well as the loss. A stateful optimizer given
  an identically zero gradient still decays its moment estimates and still steps
  from surviving momentum, so a zero-weight step is not the no-op it is under
  ``etrace_grad``; the online driver skips it entirely.
- There is no ``reduction``, because there is no accumulator to divide. Carrying
  ``etrace_grad``'s per-step weights across unchanged shrinks every update by
  the number of supervised steps rather than averaging them.

What both workflows establish
-----------------------------

- how model state is initialized and reset between sequences;
- where :func:`braintrace.compile` enters the training pipeline;
- how repeated updates are executed with compiled stateful transforms; and
- which conclusions are specific to the demonstrated task and approximation.

.. toctree::
   :hidden:
   :maxdepth: 1

   rnn_online_learning.ipynb
   snn_online_learning.ipynb
