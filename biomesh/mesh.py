# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""A module for utils for generating meshes."""

import lnmmeshio
import pathlib
import numpy as np
import meshio
import scipy.spatial as sp


def build_node_mapping(points1: np.ndarray, points2: np.ndarray) -> dict[int, int]:
    """Build a mapping from points in points2 to points in points1 using a
    tree."""
    tree1 = sp.cKDTree(points1)
    tree2 = sp.cKDTree(points2)

    result = tree1.query_ball_tree(tree2, r=1e-6)

    mapping = {}
    for i, indices in enumerate(result):
        if indices:
            assert len(indices) == 1

            mapping[indices[0]] = i

    return mapping


def _sort_cell_node_ids(arr: np.ndarray) -> tuple[int, ...]:
    """Sort cell node IDs and return as tuple."""
    return tuple(sorted([int(i) for i in arr]))


def merge_colored_stl(
    base_stl: pathlib.Path, *stl_files: pathlib.Path
) -> tuple[meshio.Mesh, list[set[int]]]:
    """Merge multiple colored STL files into a single mesh.

    Parameters
    ----------
    *stl_files : str
        Paths to the STL files to be merged.

    Returns
    -------
    tuple[meshio.Mesh, list[list[int]]]
        The merged mesh and the surface ID mapping enclosing the different volumes define by the surface ids.
    """
    base_mesh = lnmmeshio.read_mesh(str(base_stl), file_format="mimicsstl")

    # add all nodes of base mesh to new combined mesh
    nodes_list = [coords for coords in base_mesh.points]
    cell_tri_list = [cell for cell in base_mesh.cells[0].data]

    old_surface_id_mapping = {
        key: value
        for value, key in enumerate(
            np.unique(base_mesh.cell_data["medit:ref"][0]), start=1
        )
    }
    surface_id_list = [
        old_surface_id_mapping[data] for data in base_mesh.cell_data["medit:ref"][0]
    ]
    max_surface_id = max([i for i in old_surface_id_mapping.values()])

    meshes = [
        lnmmeshio.read_mesh(str(stl_file), file_format="mimicsstl")
        for stl_file in stl_files
    ]

    surface_loops = [set([id for id in surface_id_list])]

    cell_surface_ids = {
        tuple([int(i) for i in _sort_cell_node_ids(cell)]): surf_id
        for cell, surf_id in zip(cell_tri_list, surface_id_list)
    }

    for mesh_id, mesh in enumerate(meshes, start=1):
        node_mapping = build_node_mapping(np.array(nodes_list), mesh.points)

        surface_loops.append(set([]))
        new_surface_id_mapping = {}

        # match nodes or add them
        for i, coords in enumerate(mesh.points):
            if i not in node_mapping:
                # this is a new node -> add to nodes
                nodes_list.append(coords)
                node_mapping[i] = len(nodes_list) - 1

        # append cells
        for cell_id, cell in enumerate(mesh.cells[0].data):

            key = _sort_cell_node_ids([node_mapping[i] for i in cell])

            if key not in cell_surface_ids:
                # this is a new cell: Add to overall geometry
                new_cell = [node_mapping[i] for i in cell]
                cell_tri_list.append(new_cell)

                # Determine surface id
                old_sid = mesh.cell_data["medit:ref"][0][cell_id]
                if old_sid not in new_surface_id_mapping:
                    max_surface_id += 1
                    new_surface_id_mapping[old_sid] = max_surface_id

                surface_id_list.append(new_surface_id_mapping[old_sid])

                cell_surface_ids[key] = new_surface_id_mapping[old_sid]
            surface_loops[mesh_id].add(cell_surface_ids[key])

        # find intersecting line elements between each surface
        intersecting_surface_line_ids: dict[tuple, list[int]] = {}
        for i, (surf_ele, surf_id) in enumerate(zip(cell_tri_list, surface_id_list)):
            # go over each line element of the surface
            for le in [
                (surf_ele[0], surf_ele[1]),
                (surf_ele[1], surf_ele[2]),
                (surf_ele[2], surf_ele[0]),
            ]:
                id = tuple(sorted(le))

                if id in intersecting_surface_line_ids:
                    if surf_id not in intersecting_surface_line_ids[id]:
                        intersecting_surface_line_ids[id].append(surf_id)
                elif id not in intersecting_surface_line_ids:
                    intersecting_surface_line_ids[id] = [surf_id]

        line_elements = []
        line_ids = []
        used_line_ids: dict[tuple, int] = {}
        for node_ids, surf_ids in intersecting_surface_line_ids.items():
            if len(surf_ids) > 1:
                line_elements.append([node_ids[0], node_ids[1]])

                id = tuple(sorted(surf_ids))
                if id in used_line_ids:
                    line_ids.append(used_line_ids[id])
                else:
                    line_ids.append(len(used_line_ids) + 1)
                    used_line_ids[id] = len(used_line_ids) + 1

    mesh = meshio.Mesh(
        points=np.array(nodes_list),
        cells=[("triangle", cell_tri_list), ("line", line_elements)],
        cell_data={"medit:ref": [np.array(surface_id_list), np.array(line_ids)]},
    )

    # generate and write the new mesh
    return (
        mesh,
        surface_loops,
    )
