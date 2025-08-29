# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Testing mesh adaptation utilities."""
import pathlib
import meshio
import biomesh
import numpy as np


def test_lin_to_quad():
    """Testing converting linear cells to quadratic cells."""
    _my_script_dir = pathlib.Path(__file__).parent

    mesh = meshio.read(_my_script_dir / "data" / "lin_elements.vtu")
    mesh.point_data["coords"] = mesh.points

    mesh_quad = biomesh.lin_to_quad(mesh)

    np.testing.assert_allclose(mesh_quad.points, mesh_quad.point_data["coords"])

    assert len(mesh_quad.points) == 84
    assert mesh_quad.cells[0].data.shape == (3, 27)
    assert mesh_quad.cells[0].type == "hexahedron27"
    np.testing.assert_allclose(
        mesh_quad.cells[0].data[0, :],
        [
            0,
            4,
            5,
            15,
            2,
            6,
            7,
            3,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
        ],
    )

    assert mesh_quad.cells[1].data.shape == (3, 10)
    assert mesh_quad.cells[1].type == "tetra10"
    np.testing.assert_allclose(
        mesh_quad.cells[1].data[0, :],
        [16, 17, 18, 19, 70, 71, 72, 73, 74, 75],
    )
