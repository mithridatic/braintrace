"""Regression check for equilibrium observations through exposed gate states."""

import runpy

import pytest

pytest.importorskip("neuron", reason="runs only inside the NEURON image")
import unittest
from pathlib import Path


class EquilibriumObservationTest(unittest.TestCase):
    """Exercise the diagnostic against the compiled mechanism in its container."""

    def test_uses_accessible_mechanism_state(self):
        """Complete all analytic and directional checks without hidden variables."""
        runpy.run_path(str(Path(__file__).with_name("h01_pv_inactivation_equilibrium.py")))


if __name__ == "__main__":
    unittest.main()
