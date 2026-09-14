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
        "donor_accuracy", "type_coverage", "build", "driven_window", "timestep",
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


def test_the_build_term_is_the_full_population_not_the_stale_twelve():
    """The one-point SWC branch is repaired; the build term must reflect the committed audit."""
    build = {t["term"]: t for t in pl.ledger()}["build"]
    assert build["value"] == pytest.approx(1.)
    assert "12-of-104" in build["cause"]


def test_the_unqualified_terms_are_zero_and_say_why():
    terms = {t["term"]: t for t in pl.ledger()}
    for name in ("driven_window", "timestep"):
        assert terms[name]["value"] == 0.
        assert terms[name]["measured"] is False
    # the anatomy term is measured, and its reading is negative
    assert terms["anatomy_transfer"]["value"] == 0.
    assert terms["anatomy_transfer"]["measured"] is True


def test_the_product_is_zero_while_any_term_is_zero():
    assert pl.product(pl.ledger())["value_pct"] == 0.


def test_assuming_a_term_names_it_in_the_result():
    terms = pl.ledger()
    p = pl.product(terms, assume=["driven_window", "timestep", "anatomy_transfer"])
    assert p["assumptions"] == ["driven_window", "timestep", "anatomy_transfer"]
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


def test_report_quotes_the_optimistic_figure_only_with_its_assumptions():
    r = pl.report()
    assert r["cells"] == 104
    assert len(r["terms"]) == 6
    assert r["product_as_measured"]["value_pct"] == 0.
    assert r["product_if_construction_counts_as_build"]["assumptions"]
    assert "assumptions attached" in r["note"]
