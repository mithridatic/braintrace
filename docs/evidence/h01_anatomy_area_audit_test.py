"""Analytic geometry and malformed-input checks for the human anatomy audit."""

import numpy as np
import pytest

from h01_anatomy_area_audit import cable_geometry, mesh_geometry, read_swc


def test_anisotropic_units_and_radius_are_not_diameter():
    rows = read_swc("10 1 0 0 0 1000 -1\n20 3 100 0 100 1000 10\n")
    result = cable_geometry(rows, position_um=(.032, .032, .033), radius_um=.001)
    length = np.hypot(3.2, 3.3)
    assert result['length_um'] == pytest.approx(length)
    assert result['lateral_area_um2'] == pytest.approx(2 * np.pi * length)
    assert result['cross_label_edges'] == 1
    assert result['by_endpoint_labels']['1->3']['edges'] == 1


def test_frustum_slant_area_and_explicit_soma_are_separate():
    rows = read_swc("1 1 0 0 0 2 -1\n2 3 3 0 0 1 1\n")
    result = cable_geometry(rows, soma_sphere_radius_um=2)
    assert result['lateral_area_um2'] == pytest.approx(3 * np.pi * np.sqrt(10))
    assert result['soma_sphere_area_um2'] == pytest.approx(16 * np.pi)
    assert 'total_membrane_area_um2' not in result


def test_zero_length_edge_is_reported_and_not_a_disk():
    result = cable_geometry(read_swc("1 1 0 0 0 2 -1\n2 1 0 0 0 1 1\n"))
    assert result['zero_length_edges'] == 1
    assert result['lateral_area_um2'] == 0


def test_singleton_is_not_invented_cable_or_soma():
    result = cable_geometry(read_swc("1 1 0 0 0 2 -1\n"))
    assert result['edges'] == 0
    assert result['soma_sphere_area_um2'] is None


@pytest.mark.parametrize('text', [
    '', '1 1 0 0 0 1', '1 1 0 0 nan 1 -1',
    '1 1 0 0 0 0 -1', '1 1 0 0 0 -1 -1',
    '1.5 1 0 0 0 1 -1', '1 1 0 0 0 1 -1\n1 3 1 0 0 1 1',
    '1 1 0 0 0 1 -1\n2 3 1 0 0 1 9',
    '1 1 0 0 0 1 -1\n2 3 1 0 0 1 -1',
    '1 1 0 0 0 1 -1\n2 3 1 0 0 1 3\n3 3 2 0 0 1 2',
])
def test_invalid_swc_is_rejected(text):
    with pytest.raises(ValueError):
        cable_geometry(read_swc(text))


@pytest.mark.parametrize('kwargs', [
    {'position_um': (1, 0, 1)}, {'position_um': (1, 1)},
    {'radius_um': -1}, {'radius_um': np.nan}, {'soma_sphere_radius_um': 0},
])
def test_invalid_units_or_soma_are_rejected(kwargs):
    with pytest.raises(ValueError):
        cable_geometry(read_swc('1 1 0 0 0 1 -1'), **kwargs)


def test_mesh_seams_duplicates_and_open_boundary():
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [3, 2, 4], [2, 1, 0], [0, 0, 1]]
    result = mesh_geometry(vertices, faces, position_um=(2, 3, 1))
    assert result['area_um2'] == pytest.approx(6)
    assert result['duplicate_vertices'] == 1
    assert result['duplicate_faces'] == 1
    assert result['degenerate_faces'] == 1
    assert result['boundary_edges'] == 4
    assert result['components'] == 1
    assert result['component_areas_um2'] == pytest.approx([6])


def test_closed_tetrahedron_and_detached_triangle():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [10, 0, 0], [11, 0, 0], [10, 1, 0]]
    faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2], [4, 5, 6]]
    result = mesh_geometry(vertices, faces)
    assert result['area_um2'] == pytest.approx(2 + np.sqrt(3) / 2)
    assert result['boundary_edges'] == 3
    assert result['nonmanifold_edges'] == 0
    assert result['components'] == 2


def test_nonmanifold_edge_is_visible():
    result = mesh_geometry([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                            [0, -1, 0], [0, 0, 1]],
                           [[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    assert result['nonmanifold_edges'] == 1


@pytest.mark.parametrize('vertices,faces', [
    ([], []), ([[0, 0, 0]], []),
    ([[0, 0, np.nan]], [[0, 0, 0]]),
    ([[0, 0, 0]], [[0, 0, 1]]),
    ([[0, 0, 0]], [[0, 0, -1]]),
    ([[0, 0, 0]], [[0, 0, .5]]),
])
def test_invalid_mesh_is_rejected(vertices, faces):
    with pytest.raises(ValueError):
        mesh_geometry(vertices, faces)
