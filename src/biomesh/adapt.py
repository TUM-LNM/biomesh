# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Small utilities for mesh adaption (e.g. converting linear elements to
quadratic elements)"""

from typing import Callable, Optional

import meshio
import numpy as np

# Mapping from linear cell types to the node indices required for constructing quadratic elements.
# Each entry in the list corresponds to the nodes of the new element. If a node of a new element
# lies between two existing nodes, the list contains the indices of the existing nodes.
_lin_to_quad_nodes = {
    "hexahedron": [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ],
    "tetra": [
        [0],
        [1],
        [2],
        [3],
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 3],
        [2, 3],
    ],
    "wedge": [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 5],
        [5, 3],
        [0, 3],
        [1, 4],
        [2, 5],
    ],
    "triangle": [
        [0],
        [1],
        [2],
        [0, 1],
        [1, 2],
        [2, 0],
    ],
    "quad": [[0], [1], [2], [3], [0, 1], [1, 2], [2, 3], [3, 0], [0, 1, 2, 3]],
    "line": [[0], [1], [0, 1]],
}
_lin_to_quad_cell_type = {
    "hexahedron": "hexahedron27",
    "tetra": "tetra10",
    "wedge": "wedge15",
    "triangle": "triangle6",
    "quad": "quad9",
    "line": "line3",
}

_hex8_to_hex20 = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def adapt_mesh(
    mesh: meshio.Mesh,
    adapting_scheme: Callable[[str], Optional[tuple[str, list[list[int]]]]],
) -> meshio.Mesh:
    """Adapt a mesh to a higher-order (quadratic) representation using a
    provided adapting scheme.

    This function takes a mesh and, for each cell block, determines whether it should be adapted
    to a new cell type. If so, it generates new mid-edge nodes and updates
    the mesh accordingly. The adapting scheme determines the mapping from the original cell type
    to the new cell type and specifies how to construct new nodes.

    Parameters:
        mesh: The input mesh to be adapted.
        adapting_scheme:
            A function that takes a cell type string and returns either:
                - None, if the cell type should not be adapted,
                - or a tuple of (new_cell_type, node_mapping), where:
                    new_cell_type : str
                        The name of the quadratic cell type (e.g., "triangle6").
                    node_mapping : list
                        A list specifying how to map old nodes and create new mid-edge nodes.
                        Each entry is either a tuple with one index (existing node) or multiple
                        indices (to create a new node as the mean of those nodes).

    Returns:
        The adapted mesh with quadratic cells and updated points, point data, and cell data.

    Notes
        - Cell blocks that are already quadratic or are not to be adapted are left unchanged.
        - New mid-edge nodes are created only once per unique edge and reused across cells.
        - Point data is interpolated for new nodes using the mean of the corresponding data.
    """
    new_points = [coord for coord in mesh.points]
    new_point_data = {key: [d for d in data] for key, data in mesh.point_data.items()}
    new_cell_blocks = []
    new_cell_data: dict[str, list[np.ndarray]] = {
        key: [] for key in mesh.cell_data.keys()
    }

    # dictionary to keep track of new middle points, key is the node id of the two edge points
    new_middle_points: dict[tuple[int, int], int] = {}

    for i, cellblock in enumerate(mesh.cells):
        new_cells = []
        cell_type = cellblock.type

        # determine the mapping of the nodes to the new quadratic celltype
        scheme = adapting_scheme(cell_type)

        if not scheme:
            # don't adapt this block
            new_cell_blocks.append(cellblock)
            for key, data in mesh.cell_data.items():
                new_cell_data[key].append(data[i])
            continue

        new_cell_type, node_mapping = scheme

        for cell in cellblock.data:
            cell_nodes = []
            for node in node_mapping:
                if len(node) == 1:
                    # this is a node that already existed in the previous mesh
                    cell_nodes.append(cell[node[0]])
                else:
                    # this node is a new middle node

                    # we need to determine whether we have already created this middle node
                    key = tuple(sorted([cell[n] for n in node]))
                    if key not in new_middle_points:
                        # this node does not exist yet, create it
                        new_middle_points[key] = len(new_points)
                        new_points.append(
                            np.mean([mesh.points[cell[n]] for n in node], axis=0)
                        )
                        for name in mesh.point_data.keys():
                            new_point_data[name].append(
                                np.mean(
                                    [mesh.point_data[name][cell[n]] for n in node],
                                    axis=0,
                                    dtype=mesh.point_data[name].dtype,
                                )
                            )

                    cell_nodes.append(new_middle_points[key])

            new_cells.append(cell_nodes)

        new_cell_blocks.append(
            meshio.CellBlock(
                cell_type=new_cell_type,
                data=np.array(new_cells, dtype=np.int64),
            )
        )
        for key, data in mesh.cell_data.items():
            new_cell_data[key].append(data[i])

    return meshio.Mesh(
        points=np.array(new_points, dtype=np.float64),
        cells=new_cell_blocks,
        point_data={key: np.array(data) for key, data in new_point_data.items()},
        cell_data={
            key: [np.array(d) for d in data] for key, data in new_cell_data.items()
        },
    )


def lin_to_quad(mesh: meshio.Mesh) -> meshio.Mesh:
    """Convert linear elements to quadratic elements in a mesh.

    This function returns a new mesh in which all linear elements (triangles, quadrilaterals, tetrahedra, and hexahedra)
    are converted into their corresponding quadratic elements.

    Args:
        mesh:
            The input mesh containing linear elements.

    Returns:
        The modified mesh with quadratic elements.
    """

    def lin_to_quad(cell_type: str) -> Optional[tuple[str, list[list[int]]]]:
        """Returns the node mapping for quadratic elements."""
        if cell_type in ["vertex", "hexahedron27", "tetra10", "triangle6", "quad9"]:
            return None

        return _lin_to_quad_cell_type[cell_type], _lin_to_quad_nodes[cell_type]

    return adapt_mesh(mesh, lin_to_quad)


def to_serendipity(mesh: meshio.Mesh) -> meshio.Mesh:
    """Converts elements to serendipity elements in a mesh.

    Args:
        mesh:
            The input mesh containing linear elements.

    Returns:
        The modified mesh with serendipity elements.
    """

    def _to_serendipity(cell_type: str) -> Optional[tuple[str, list[list[int]]]]:
        """Returns the node mapping for serendipity elements."""
        if cell_type in ["hexahedron20"]:
            return None

        if cell_type not in ["hexahedron"]:
            raise ValueError(
                f"Cell type {cell_type} not supported for serendipity conversion."
            )

        return "hexahedron20", _hex8_to_hex20

    return adapt_mesh(mesh, _to_serendipity)
