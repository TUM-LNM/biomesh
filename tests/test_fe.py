# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Tests for small finite element ulitities."""

import pathlib

import meshio
import numpy as np
import pytest

import biomesh


@pytest.mark.parametrize("shape", ["tetra", "tetra10"])
def test_node_ordering_tet(shape):
    """Test node ordering for tetrahedral elements."""
    mesh_tet = meshio.Mesh(
        points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        cells=[("tetra", [[0, 1, 2, 3]])],
    )

    if shape == "tetra10":
        mesh_tet = biomesh.lin_to_quad(mesh_tet)

    ref_points = mesh_tet.points[mesh_tet.cells_dict[shape]].reshape((-1, 3))

    np.testing.assert_allclose(ref_points, biomesh.fe.ref_coords(shape))


@pytest.mark.parametrize("shape", ["hexahedron"])
def test_node_ordering_hex(shape):
    """Test node ordering for hexahedral elements."""
    mesh_tet = meshio.Mesh(
        points=[
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        cells=[("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]])],
    )

    ref_points = mesh_tet.points[mesh_tet.cells_dict[shape]].reshape((-1, 3))

    np.testing.assert_allclose(ref_points, biomesh.fe.ref_coords(shape))


@pytest.mark.parametrize("shape", ["hex", "tet", "tet10"])
def test_grad(shape):
    """Test computing the gradient of a scalar field."""
    _my_script_dir = pathlib.Path(__file__).parent

    mesh = meshio.read(_my_script_dir / "data" / f"test_strip_{shape}.vtu")

    phi = mesh.points[:, 0]

    grad = biomesh.fe.grad(mesh, phi)

    assert grad.shape == (mesh.points.shape[0], 3)

    assert grad[:, 0] == pytest.approx(1.0)
    assert grad[:, 1] == pytest.approx(0.0)
    assert grad[:, 2] == pytest.approx(0.0)
