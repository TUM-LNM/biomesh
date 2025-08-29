# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""A module to merge multuple meshes into one larger mesh."""

import meshio
import numpy as np


def merge(*meshes: meshio.Mesh) -> meshio.Mesh:
    """Merge multiple meshio.Mesh objects into a single mesh.

    This function combines the points, cells, point data, and cell data from all input meshes
    into a new meshio.Mesh object. It handles the necessary index adjustments for points and
    cells, and ensures that point and cell data are correctly merged and aligned.

    Args:
        *meshes: Variable number of meshio.Mesh objects to merge.

    Returns:
        A new meshio.Mesh object containing the merged data from all input meshes.

    Notes:
        - Point and cell data arrays are padded with zeros if not present at other meshes.
    """
    new_points = np.vstack([m.points for m in meshes])
    point_data: dict[str, np.ndarray] = {}
    cell_blocks = []
    cell_data: dict[str, list[np.ndarray]] = {}

    add_to_node_ids = 0
    mesh_cell_block_offset = 0

    def make_zero_cell_data(shape: str) -> list[np.ndarray]:
        """Create zero-initialized cell data arrays for the combined mesh."""
        empty_data = []
        for m in meshes:
            for block in m.cells:
                empty_data.append(np.zeros((len(block.data), *shape)))

        return empty_data

    for mesh in meshes:
        # transfer point data
        for key, data in mesh.point_data.items():
            if key not in point_data:
                point_data[key] = np.zeros((len(new_points), *data.shape[1:]))

            point_data[key][add_to_node_ids : add_to_node_ids + len(mesh.points)] = data

        # add cell blocks
        for block in mesh.cells:
            cell_blocks.append(
                meshio.CellBlock(block.type, block.data + add_to_node_ids)
            )

        # add cell data
        for key, data in mesh.cell_data.items():
            if key not in cell_data:
                cell_data[key] = make_zero_cell_data(data[0].shape[1:])

            for i, d in enumerate(data):
                cell_data[key][mesh_cell_block_offset + i] = d

        add_to_node_ids += len(mesh.points)
        mesh_cell_block_offset += len(mesh.cells)

    return meshio.Mesh(
        points=new_points, cells=cell_blocks, point_data=point_data, cell_data=cell_data
    )
