# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Test merge meshes."""

import pathlib

import meshio
import numpy as np

import biomesh


def test_merge():
    """Test merging multiple meshes."""
    script_dir = pathlib.Path(__file__).parent
    meshes = [
        meshio.read(str(script_dir / "data" / "test_strip_hex.vtu")),
        meshio.read(str(script_dir / "data" / "test_strip_tet.vtu")),
        meshio.read(str(script_dir / "data" / "test_strip_tet10.vtu")),
    ]

    # add data to the meshes
    for i, mesh in enumerate(meshes):
        mesh.point_data["coords"] = mesh.points
        mesh.point_data[f"node_id_{i}"] = np.arange(len(mesh.points))

        mesh.cell_data[f"cell_id_{i}"] = [np.arange(len(mesh.cells[0].data))]

    full_mesh = biomesh.merge(*meshes)

    assert len(full_mesh.points) == sum([len(m.points) for m in meshes])
    assert len(full_mesh.cells) == 3

    for cell_block, mesh in zip(full_mesh.cells, meshes):
        assert len(cell_block.data) == len(mesh.cells[0].data)
        for cell_full, cell_ref in zip(cell_block.data, mesh.cells[0].data):
            np.testing.assert_allclose(
                full_mesh.points[cell_full], mesh.points[cell_ref]
            )

    # assert all point data
    np.testing.assert_allclose(full_mesh.point_data["coords"], full_mesh.points)

    node_id_offset = 0
    for i, mesh in enumerate(meshes):
        np.testing.assert_allclose(
            full_mesh.point_data[f"node_id_{meshes.index(mesh)}"][
                node_id_offset : node_id_offset + len(mesh.points)
            ],
            np.arange(len(mesh.points)),
        )
        node_id_offset += len(mesh.points)

        for j in range(len(meshes)):
            full_mesh.cell_data[f"cell_id_{i}"][j] = (
                np.arange(len(mesh.cells[0].data)) * i == j
            )
