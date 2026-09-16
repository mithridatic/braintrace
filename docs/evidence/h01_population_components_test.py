import json
import zipfile

import pytest

from docs.evidence import h01_population_components as module


def _swc(codes):
    return "\n".join(f"{i} {c} {i}.0 0.0 0.0 400.0 {i-1}" for i, c in enumerate(codes)).encode()


@pytest.fixture
def archive(tmp_path):
    path = tmp_path/"cells.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("10.0.swc", _swc([3, 3, 1, 1, 0]))
        z.writestr("10.1.swc", _swc([0, 0]))
        z.writestr("11.0.swc", _swc([0, 0, 0, 0, 0, 0]))
        z.writestr("11.1.swc", _swc([3, 5]))
        z.writestr("README.txt", b"ignored")
    return path


def test_component_record_counts_nodes_and_codes():
    record = module.component_record("10.0.swc", _swc([3, 3, 1, 5]))
    assert record == {"cell_id": "10", "component": 0, "nodes": 4, "soma_nodes": 2, "ais_nodes": 1}


def test_inventory_and_per_cell_flag_fragments_and_somaless_largest(archive):
    rows = module.per_cell(module.inventory(archive))
    assert [r["cell_id"] for r in rows] == ["10", "11"]
    assert rows[0]["components"] == 2 and rows[0]["soma_components"] == [0] and rows[0]["largest_has_soma"]
    assert rows[0]["fragments_without_soma"] == 1 and rows[0]["largest_share"] == pytest.approx(5/7)
    assert rows[1]["largest_component"] == 0 and not rows[1]["largest_has_soma"]


def test_contact_check_and_summary(archive):
    rows = module.per_cell(module.inventory(archive))
    network = {"contacts": [{"annotation_id": "1", "pre_cell": "10", "post_cell": "11",
                             "pre_placement": {"component": 0}, "post_placement": {"component": 0}}]}
    checks = module.contact_check(network, rows)
    assert [c["on_soma_component"] for c in checks] == [True, False]
    summary = module.summarize(rows, checks)
    assert summary["cells"] == 2 and summary["components"] == 4
    assert summary["cells_where_largest_lacks_soma"] == 1 and summary["contact_endpoints_on_soma_component"] == 1


def test_main_writes_pages(archive, tmp_path):
    network = tmp_path/"network.json"
    network.write_text(json.dumps({"contacts": []}))
    module.main(["--archive", str(archive), "--network", str(network), "--output", str(tmp_path/"out")])
    assert (tmp_path/"out.json").exists()
    assert "| 10 | 2 | 7 | [0] |" in (tmp_path/"out.md").read_text()


def test_main_without_a_network_records_no_contacts(archive, tmp_path):
    module.main(["--archive", str(archive), "--no-network", "--output", str(tmp_path/"out")])
    report = json.loads((tmp_path/"out.json").read_text())
    assert report["contacts"] == [] and report["summary"]["contact_endpoints"] == 0
    assert [c["cell_id"] for c in report["cells"]] == ["10", "11"]
