# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Small utilities for mesh adaption (e.g. converting linear elements to
quadratic elements)"""

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
    "triangle": "triangle6",
    "quad": "quad9",
    "line": "line3",
}


def lin_to_quad(mesh: meshio.Mesh) -> meshio.Mesh:
    """Convert linear elements to quadratic elements in a mesh.

    This function returns a new mesh in which all linear elements (triangles, quadrilaterals, tetrahedra, and hexahedra)
    are converted into their corresponding quadratic elements.

    Args:
    -------
    mesh: meshio.Mesh
        The input mesh containing linear elements.

    Returns:
        meshio.Mesh: The modified mesh with quadratic elements.
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

        if cell_type in ["vertex"]:
            # vertices do not need to be converted
            continue

        if cell_type in ["hexahedron27", "tetra10", "triangle6", "quad9"]:
            # this cell block is already quadratic, no need to convert
            new_cell_blocks.append(cellblock)
            for key, data in mesh.cell_data.items():
                new_cell_data[key].append(data[i])
            continue

        # determine the mapping of the nodes to the new quadratic celltype
        node_mapping = _lin_to_quad_nodes[cell_type]
        new_cell_type = _lin_to_quad_cell_type[cell_type]

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
