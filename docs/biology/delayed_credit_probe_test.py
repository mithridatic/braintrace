"""Closed-form direct, delayed, warm-up and zero-decay credit controls."""

import brainstate
import pytest
from docs.biology.delayed_credit_probe import measure, validate


@pytest.mark.parametrize('delay,emission', [(0, 0), (1, 0), (5, 0), (5, 2)])
@pytest.mark.parametrize('decay', [0., .2, .8, .99])
def test_closed_form_delayed_credit(delay, emission, decay):
    with brainstate.environ.context(precision=64):
        validate(measure(delay, emission, decay))


@pytest.mark.parametrize('args', [(-1, 0, .8), (1, -1, .8), (1, 0, 1.)])
def test_invalid_trace_domain(args):
    with pytest.raises(ValueError):
        measure(*args)
