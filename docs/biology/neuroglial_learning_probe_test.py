"""Full synthetic neuroglial finite-window oracle acceptance."""

from pathlib import Path
import hashlib
import json

from examples.pp_prop.h01_neuroglial_test import configured, imported, precision
from examples.pp_prop.h01_session import H01Session, numerical_settings
from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter
from docs.biology import neuroglial_learning_probe as probe
from docs.biology.neuroglial_learning_probe import qualify, validate


def test_finite_window_encoder_gradient_and_descent(imported, tmp_path):
    topology, archive, biology, assets = configured(imported, tmp_path)
    settings = numerical_settings()
    settings['biology'] = biology
    trainer = Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
    session = H01Session.build(topology, archive, trainer, settings=settings, biology_assets=assets)
    original = session.model.input_weight.value.copy()
    report = qualify(session)
    validate(report)
    assert (session.model.input_weight.value == original).all()
    assert session.model.stepper.tick.value == 0
    assert all((factor == 0).all() for factor in session.learner.factors.value)


def test_evidence_records_source_identity_and_passed_metrics(tmp_path):
    probe.main(tmp_path)
    report = json.loads((tmp_path/'learning-probe.json').read_text())
    settings = json.loads((tmp_path/'settings.json').read_text())
    validate(report)
    assert report['probe_sha256'] == hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
    assert settings['biology']['extracellular']['origin'] == 'synthetic'
    assert settings['dt_ms'] == report['dt_ms']
    assert settings['implementation_sha256'] == numerical_settings()['implementation_sha256']
