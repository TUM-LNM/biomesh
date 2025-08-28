# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""A module for reordering nodes in a finite element mesh."""

import meshio
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import copy


def reverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Compute the inverse of a permutation array."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def build_csr_from_multiple_elements(
    n_nodes: int, *elements_lists: np.ndarray
) -> sp.csr_array:
    """Build a CSR adjacency matrix from multiple lists of cell connectivity.

    n_nodes: total number of nodes
    elements_lists: variable number of arrays/lists of elements
    """
    adj = defaultdict(set)

    for elements in elements_lists:
        for elem in elements:
            for i, elem_i in enumerate(elem):
                for j, elem_j in enumerate(elem):
                    if i != j:
                        adj[elem_i].add(elem_j)

    # Convert adjacency to CSR arrays
    data = []
    indices = []
    indptr = [0]

    for node in range(n_nodes):
        neighbors = sorted(adj[node])  # optional: sort neighbors
        indices.extend(neighbors)
        data.extend([1] * len(neighbors))
        indptr.append(len(indices))

    return sp.csr_array((data, indices, indptr), shape=(n_nodes, n_nodes))


def reorder(mesh: meshio.Mesh) -> None:
    """Reorder the nodes of a mesh in-place to minimize the bandwidth of the
    adjacency matrix.

    This function uses the reverse Cuthill-McKee algorithm to find an optimal ordering
    of the nodes in the mesh, which is particularly useful for sparse matrices.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh to be reordered.
    """
    csr_array = build_csr_from_multiple_elements(
        len(mesh.points), *mesh.cells_dict.values()
    )

    # compute optimal node ordering for each independent block of the mesh
    n_components, labels = sp.csgraph.connected_components(csr_array, directed=False)
    perm_global = []
    for comp in range(n_components):
        nodes_in_comp = (labels == comp).nonzero()[0]
        csr_array_sub = csr_array[nodes_in_comp, :][:, nodes_in_comp]
        perm_sub = sp.csgraph.reverse_cuthill_mckee(csr_array_sub)
        perm_global.extend(nodes_in_comp[perm_sub])

    points_permutation = np.array(perm_global)
    reverse_points_permutation = reverse_permutation(points_permutation)

    # reorder the mesh points
    new_points = mesh.points[points_permutation]
    new_point_data = {}

    # reorder point data
    for key, data in mesh.point_data.items():
        new_point_data[key] = data[points_permutation]

    cell_blocks = []

    # adapt cell connectivity
    for cell_block in mesh.cells:
        cell_blocks.append(
            meshio.CellBlock(
                cell_type=cell_block.type,
                data=reverse_points_permutation[cell_block.data],
            )
        )

    return meshio.Mesh(
        points=new_points,
        point_data=new_point_data,
        cells=cell_blocks,
        cell_data=copy.deepcopy(mesh.cell_data),
        field_data=copy.deepcopy(mesh.field_data),
    )
