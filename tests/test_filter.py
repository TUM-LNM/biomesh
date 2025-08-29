# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Testing filtering meshes."""

import pathlib

import meshio
import numpy as np

import biomesh


def test_filter_by_cellblock():
    """Test the filter by a specific cellblock filter."""
    script_dir = pathlib.Path(__file__).parent
    meshes = [
        meshio.read(str(script_dir / "data" / "test_strip_hex.vtu")),
        meshio.read(str(script_dir / "data" / "test_strip_tet.vtu")),
        meshio.read(str(script_dir / "data" / "test_strip_tet10.vtu")),
    ]

    full_mesh = biomesh.merge(*meshes)
    full_mesh.point_data["coords"] = full_mesh.points

    # filter tetra10 cells
    full_mesh_tet10 = biomesh.filter_by_cellblock(
        full_mesh, lambda block: block.type == "tetra10"
    )

    np.testing.assert_allclose(full_mesh_tet10.points, meshes[2].points)

    assert len(full_mesh_tet10.cells) == 1
    assert full_mesh_tet10.cells[0].type == "tetra10"
    np.testing.assert_allclose(full_mesh_tet10.cells[0].data, meshes[2].cells[0].data)

    np.testing.assert_allclose(
        full_mesh_tet10.point_data["coords"], full_mesh_tet10.points
    )


def test_filter_by_cellblock_point_mapping():
    """Test the points map by a specific cellblock filter."""
    script_dir = pathlib.Path(__file__).parent
    meshes = [
        meshio.read(str(script_dir / "data" / "test_strip_hex.vtu")),
        meshio.read(str(script_dir / "data" / "test_strip_tet.vtu")),
        meshio.read(str(script_dir / "data" / "test_strip_tet10.vtu")),
    ]

    full_mesh = biomesh.merge(*meshes)
    full_mesh.point_data["coords"] = full_mesh.points

    # filter tetra10 cells
    map = biomesh.filter_by_cellblock_point_mapping(
        full_mesh, lambda block: block.type == "tetra10"
    )

    np.testing.assert_allclose(map[full_mesh.cells[2].data], meshes[2].cells[0].data)
