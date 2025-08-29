# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Filter for meshes to extract a subset."""

from typing import Callable

import meshio
import numpy as np


def filter_by_cellblock_point_mapping(
    mesh: meshio.Mesh, filter: Callable[[meshio.CellBlock], bool]
) -> np.ndarray:
    """Generates a mapping from original mesh point indices to new indices
    based on selected cell blocks.

    This function filters the cell blocks of a mesh using a provided filter function, collects all unique point indices
    referenced by the filtered cell blocks, and creates a mapping array. The mapping array has the same length as the
    original number of points in the mesh. For points included in the filtered cell blocks, the array contains their new
    consecutive indices; for points not included, the value is -1.

    Args:
        mesh: The input mesh object containing points and cell blocks.

        filter: A function that takes a cell block and returns True if it should be included.

    Returns:
        An array of length `len(mesh.points)` where each entry is the new index of the point if it is included
        in the filtered cell blocks, or -1 otherwise.
    """

    cell_filter = np.array(
        [
            cell_block_id
            for cell_block_id, cell_block in enumerate(mesh.cells)
            if filter(cell_block)
        ]
    )

    all_remaining_points = np.unique(
        np.concatenate(
            tuple(
                [
                    mesh.cells[cell_block_id].data.flatten()
                    for cell_block_id in cell_filter
                ]
            )
        )
    )

    point_filter = np.where(np.isin(np.arange(len(mesh.points)), all_remaining_points))[
        0
    ]
    point_id_map = np.zeros(len(mesh.points), dtype=int) - 1
    point_id_map[point_filter] = np.arange(len(point_filter))

    return point_id_map


def filter_by_cellblock(
    mesh: meshio.Mesh, filter: Callable[[meshio.CellBlock], bool]
) -> meshio.Mesh:
    """Filter a mesh to include only cells of specified types.

    Args:
        mesh: The input mesh to filter.
        filter: A function that takes a cell block and returns True if it should be included.

    Returns:
        A new meshio.Mesh object containing only the cells of the specified types,
        with unused points removed and point/cell data filtered accordingly.

    Notes
    - If no cells of the specified types are found, returns an empty mesh with zero points and cells.
    """

    # return only the cells of the given celltypes
    cell_filter = np.array(
        [
            cell_block_id
            for cell_block_id, cell_block in enumerate(mesh.cells)
            if filter(cell_block)
        ]
    )

    return filter_by_block_ids(mesh, cell_filter)


def filter_by_block_ids(mesh: meshio.Mesh, cellblock_ids: np.ndarray) -> meshio.Mesh:
    """Extracts a subset of a mesh based on specified cell block indices.

    Given a meshio.Mesh object and an array of cell block indices, this function creates a new mesh containing only the cells from the specified cell blocks and the points referenced by those cells. The point indices are remapped so that the resulting mesh is consistent and compact. Associated point and cell data are also filtered accordingly.

    Args:
        mesh: The input mesh from which to extract cell blocks.
        cellblock_ids: An array of indices specifying which cell blocks to include in the output mesh.

    Returns:
        A new meshio.Mesh object containing only the specified cell blocks and their associated points and data.

    Notes:
        - If cellblock_ids is empty, returns an empty mesh with no points or cells.
        - The function remaps point indices so that the output mesh is self-contained.
    """
    if len(cellblock_ids) == 0:
        return meshio.Mesh(points=np.zeros((0, 3)), cells=[])

    all_remaining_points = np.unique(
        np.concatenate(
            tuple(
                [
                    mesh.cells[cell_block_id].data.flatten()
                    for cell_block_id in cellblock_ids
                ]
            )
        )
    )

    point_filter = np.where(np.isin(np.arange(len(mesh.points)), all_remaining_points))[
        0
    ]
    point_id_map = np.zeros(len(mesh.points), dtype=int) - 1
    point_id_map[point_filter] = np.arange(len(point_filter))

    # now we have to extract all remaining nodes, cellblocks and the corresponding data
    return meshio.Mesh(
        points=mesh.points[point_filter],
        cells=[
            meshio.CellBlock(mesh.cells[id].type, point_id_map[mesh.cells[id].data])
            for id in cellblock_ids
        ],
        point_data={key: data[point_filter] for key, data in mesh.point_data.items()},
        cell_data={
            key: [data[id] for id in cellblock_ids]
            for key, data in mesh.cell_data.items()
        },
    )
