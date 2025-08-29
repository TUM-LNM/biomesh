# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""A test suite for the dummy Laplace solver."""

import pathlib

import meshio
import numpy as np
import pytest

import biomesh


def test_cell_stiffness_hex():
    """Tests the computation of the cell stiffness matrix for hexahedron
    elements."""
    ele_stiffness = biomesh.laplace.get_cell_stiffness(
        "hexahedron",
        np.array(
            [
                [0.01, 0.02, 0.03],
                [1.04, 0.05, 0.06],
                [1.07, 1.08, 0.09],
                [0.10, 1.11, 0.12],
                [0.13, 0.14, 1.15],
                [1.16, 0.17, 1.18],
                [1.19, 1.20, 1.21],
                [0.22, 1.23, 1.24],
            ]
        ),
    )

    cell_stiffness_ref = np.array(
        [
            [
                0.29880176,
                -0.02981328,
                -0.09233284,
                -0.00447429,
                0.00964636,
                -0.06895349,
                -0.06358174,
                -0.04929248,
            ],
            [
                -0.02981328,
                0.36072577,
                0.00466812,
                -0.10708496,
                -0.10877225,
                0.0238554,
                -0.05524495,
                -0.08833386,
            ],
            [
                -0.09233284,
                0.00466812,
                0.3942286,
                -0.01050138,
                -0.10833386,
                -0.10543891,
                0.0229797,
                -0.10526942,
            ],
            [
                -0.00447429,
                -0.10708496,
                -0.01050138,
                0.38106476,
                -0.10860275,
                -0.09833386,
                -0.07558393,
                0.02351642,
            ],
            [
                0.00964636,
                -0.10877225,
                -0.10833386,
                -0.10860275,
                0.397687,
                -0.00212362,
                -0.09220777,
                0.01270689,
            ],
            [
                -0.06895349,
                0.0238554,
                -0.10543891,
                -0.09833386,
                -0.00212362,
                0.37085085,
                -0.01289647,
                -0.10695989,
            ],
            [
                -0.06358174,
                -0.05524495,
                0.0229797,
                -0.07558393,
                -0.09220777,
                -0.01289647,
                0.31409264,
                -0.03755748,
            ],
            [
                -0.04929248,
                -0.08833386,
                -0.10526942,
                0.02351642,
                0.01270689,
                -0.10695989,
                -0.03755748,
                0.35118983,
            ],
        ]
    )

    np.testing.assert_allclose(ele_stiffness, cell_stiffness_ref, rtol=1e-5)


def test_cell_stiffness_tet():
    """Tests the computation of the cell stiffness matrix for tetrahedral
    elements."""
    ele_stiffness = biomesh.laplace.get_cell_stiffness(
        "tetra",
        np.array(
            [
                [0.01, 0.02, 0.03],
                [1.04, 0.05, 0.06],
                [1.07, 1.08, 0.09],
                [0.10, 1.11, 0.12],
            ]
        ),
    )

    cell_stiffness_ref = np.array(
        [
            [2.78277778, -2.95444444, 3.11611111, -2.94444444],
            [-2.95444444, 3.14611111, -3.31777778, 3.12611111],
            [3.11611111, -3.31777778, 3.50944444, -3.30777778],
            [-2.94444444, 3.12611111, -3.30777778, 3.12611111],
        ]
    )

    np.testing.assert_allclose(ele_stiffness, cell_stiffness_ref, rtol=1e-5)


@pytest.mark.parametrize("shape", ["hex", "tet", "tet10"])
def test_solve_laplace(shape):
    """Tests solving the Laplace equation on various meshes of different cell
    type."""
    _my_script_dir = pathlib.Path(__file__).parent

    mesh = meshio.read(_my_script_dir / "data" / f"test_strip_{shape}.vtu")

    left_points = np.argwhere(mesh.points[:, 0] == 0.0).flatten()
    right_points = np.argwhere(mesh.points[:, 0] == 3.0).flatten()

    dbc_points = np.concatenate((left_points, right_points))
    dbc_values = np.concatenate(
        (np.zeros(left_points.shape[0]), np.ones(right_points.shape[0]))
    )

    result = biomesh.laplace.solve(mesh, dbc_points, dbc_values)

    result_ref = mesh.points[:, 0] / 3.0

    np.testing.assert_allclose(result, result_ref, rtol=1e-5)


@pytest.mark.parametrize("shape", ["hex", "tet", "tet10"])
def test_solve_onezero(shape):
    """Test the dummy Laplace solver with onezero boundary conditions."""
    _my_script_dir = pathlib.Path(__file__).parent

    mesh = meshio.read(_my_script_dir / "data" / f"test_strip_{shape}.vtu")

    left_points = np.argwhere(mesh.points[:, 0] == 0.0).flatten()
    right_points = np.argwhere(mesh.points[:, 0] == 3.0).flatten()

    result = biomesh.laplace.solve_onezero(mesh, right_points, left_points)

    result_ref = mesh.points[:, 0] / 3.0

    np.testing.assert_allclose(result, result_ref, rtol=1e-5)
