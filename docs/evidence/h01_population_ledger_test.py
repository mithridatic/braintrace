"""Tests for the population accuracy ledger.

The property these tests exist to protect is the one the stage-15/16 scorer got wrong once: a
term with no evidence must score zero and stay in the product, never be dropped, because dropping
it raises the score of exactly the case that fails it.
"""

import json

import pytest

import h01_population_ledger as pl


def test_every_term_is_present_and_named():
    """Six terms, in order, each with a value, a statement, a cause and its evidence files."""
    terms = pl.ledger()
    assert [t["term"] for t in terms] == [
        "donor_accuracy", "type_coverage", "construction", "driven_window", "timestep",
        "anatomy_transfer"]
    for t in terms:
        assert 0. <= t["value"] <= 1.
        assert t["statement"] and t["cause"]
        assert t["evidence"]


def test_the_measured_terms_carry_the_committed_numbers():
    terms = {t["term"]: t for t in pl.ledger()}
    acc = json.load(open(pl.HERE / pl.ACCURACY, encoding="utf-8"))
    types = json.load(open(pl.HERE / pl.TYPES, encoding="utf-8"))
    assert terms["donor_accuracy"]["value"] == pytest.approx(acc["b3"]["mean_pct"] / 100.)
    assert terms["type_coverage"]["value"] == pytest.approx(types["summary"]["matched"] / 104.)


def test_the_construction_term_is_the_full_population_not_the_stale_twelve():
    """The one-point SWC branch is repaired; the build term must reflect the committed audit."""
    build = {t["term"]: t for t in pl.ledger()}["construction"]
    assert build["value"] == pytest.approx(1.)
    assert "12-of-104" in build["cause"]


def test_the_unqualified_terms_are_zero_and_say_why(without_vast_decisions):
    terms = {t["term"]: t for t in pl.ledger()}
    assert terms["driven_window"]["value"] == 0.
    assert terms["driven_window"]["measured"] is False
    # The historical negative interpretation was invalidated by the converter audit.
    assert terms["anatomy_transfer"]["value"] == 0.
    assert terms["anatomy_transfer"]["measured"] is False
    assert 'conversion' in terms['anatomy_transfer']['cause']


def test_the_timestep_term_closed_on_the_recovered_ladder():
    """It was 0.000 'traces deleted'; the traces were recovered and the ladder qualifies a step."""
    ts = {t["term"]: t for t in pl.ledger()}["timestep"]
    assert ts["value"] == 1.
    assert ts["measured"] is True
    assert "0.000625" in ts["statement"]


def test_the_product_is_zero_while_any_term_is_zero(without_vast_decisions):
    assert pl.product(pl.ledger())["value_pct"] == 0.


def test_assuming_a_term_names_it_in_the_result():
    terms = pl.ledger()
    p = pl.product(terms, assume=["driven_window", "anatomy_transfer"])
    assert p["assumptions"] == ["driven_window", "anatomy_transfer"]
    assert {f["term"] for f in p["factors"] if f["assumed"]} == set(p["assumptions"])
    donor = next(t for t in terms if t["term"] == "donor_accuracy")["value"]
    cover = next(t for t in terms if t["term"] == "type_coverage")["value"]
    assert p["value_pct"] == pytest.approx(100. * donor * cover)


def test_a_term_cannot_be_dropped_by_assuming_it_away_silently():
    """Assuming every term gives 100 percent, and the result still lists all six as assumed."""
    terms = pl.ledger()
    p = pl.product(terms, assume=[t["term"] for t in terms])
    assert p["value_pct"] == pytest.approx(100.)
    assert len(p["factors"]) == len(terms)
    assert all(f["assumed"] for f in p["factors"])


def test_report_quotes_the_optimistic_figure_only_with_its_assumptions(without_vast_decisions):
    r = pl.report()
    assert r["cells"] == 104
    assert len(r["terms"]) == 6
    assert r["product_as_measured"]["value_pct"] == 0.
    assert r["product_if_construction_counts_as_simulation"]["assumptions"]
    assert "assumptions attached" in r["note"]


@pytest.fixture
def without_vast_decisions(monkeypatch):
    monkeypatch.setattr(pl, '_exists', lambda name: False)


def test_driven_and_transfer_terms_read_the_vast_decisions(monkeypatch):
    original = pl._read
    decisions = {
        pl.DRIVEN_DECISION: {"verdict": "PASS", "runtime": {"ei": {"status": "passed"}},
                             "controls": {"status": "passed"},
                             "refinement": {"status": "passed", "worst_voltage_mV": .4, "worst_event_ms": 0.}},
        pl.TRANSFER_DECISION: {"anatomy_transfer": .25, "verdict": "1 of 4",
                               "donors": {"l4-pyramidal-x": {"count": 13, "human_count": 12}}},
    }
    monkeypatch.setattr(pl, '_exists', lambda name: name in decisions)
    monkeypatch.setattr(pl, '_read', lambda name: decisions.get(name) or original(name))
    driven, anatomy = pl.driven_term(), pl.anatomy_term()
    assert driven["value"] == 1. and driven["measured"] is True and pl.DRIVEN_DECISION in driven["evidence"]
    assert anatomy["value"] == .25 and anatomy["measured"] is True and "13 vs human 12" in anatomy["statement"]
    decisions[pl.DRIVEN_DECISION]["verdict"] = "FAIL"
    assert pl.driven_term()["value"] == 0.


def test_invalid_correction_cannot_change_the_evidence_interpretation(monkeypatch):
    monkeypatch.setattr(pl, '_read', lambda name: {
        'status': 'failed', 'historical_transfer_interpretation': 'invalidated_by_conversion_defect'})
    with pytest.raises(ValueError, match='does not support'):
        pl.anatomy_term()


def test_cli_retains_all_six_terms_and_unavailable_transfer(tmp_path, monkeypatch, without_vast_decisions):
    original = pl.HERE
    monkeypatch.setattr(pl, '_read', lambda name: json.loads((original / name).read_text()))
    monkeypatch.setattr(pl, 'HERE', tmp_path)
    pl.main()
    record = json.loads((tmp_path / 'h01-population-accuracy-ledger.json').read_text())
    assert len(record['terms']) == 6
    assert record['terms'][-1]['measured'] is False
    assert record['product_as_measured']['value_pct'] == 0
