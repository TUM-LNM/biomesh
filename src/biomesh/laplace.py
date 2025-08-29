# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""A simple dummy Laplace-solver."""

import meshio
import numpy as np
import scipy.sparse

from . import fe

_OPTIMAL_NUMBER_OF_INTEGRATION_POINTS = {
    "hexahedron": 8,
    "tetra": 4,
    "tetra10": 4,
}


def get_cell_stiffness(cell_type: str, nodal_coords: np.ndarray) -> np.ndarray:
    """Computes the element stiffness matrix for a finite element cell for a
    simple Laplace problem.

    Args:
        cell_type: The type of the finite element cell (e.g., 'triangle', 'tetrahedron').
        nodal_coords: An array of shape (num_nodes, dim) containing the coordinates of the cell's nodes.

    Returns:
        The element stiffness matrix of shape (num_nodes, num_nodes) for the given cell.
    """

    num_nodes = nodal_coords.shape[0]

    ele_stiffness = np.zeros((num_nodes, num_nodes))

    for int_point, int_weight in zip(
        *fe.int_points_weights(
            cell_type, _OPTIMAL_NUMBER_OF_INTEGRATION_POINTS[cell_type]
        )
    ):
        derivs = fe.shape_function_first_derivatives(cell_type, int_point).T

        invJ = np.linalg.inv(derivs.dot(nodal_coords))
        detJ = 1.0 / np.linalg.det(invJ)

        dAndX = invJ.dot(derivs)

        ele_stiffness += int_weight * dAndX.T.dot(dAndX) * detJ

    return ele_stiffness


def get_global_stiffness(mesh: meshio.Mesh) -> scipy.sparse.csc_array:
    """Assemble the global stiffness matrix for a dummy Laplace-problem on a
    given mesh."""

    num_nod = mesh.points.shape[0]

    # build sparse matrix
    rows = []
    cols = []
    data = []

    for cell_block in mesh.cells:
        for cell in cell_block.data:
            ele_stiffness = get_cell_stiffness(cell_block.type, mesh.points[cell, :])

            for i in range(ele_stiffness.shape[0]):
                for j in range(ele_stiffness.shape[1]):
                    rows.append(cell[i])
                    cols.append(cell[j])
                    data.append(ele_stiffness[i, j])

    return scipy.sparse.coo_matrix(
        (
            np.array(data, dtype=float),
            (np.array(rows, dtype=int), np.array(cols, dtype=int)),
        ),
        shape=[num_nod, num_nod],
    ).tocsc()


def solve_onezero(
    mesh: meshio.Mesh, one_nodes: np.ndarray, zero_nodes: np.ndarray
) -> np.ndarray:
    """Solves a Laplace problem on the given mesh with Dirichlet boundary
    conditions set to 1 on `one_nodes` and 0 on `zero_nodes`.

    Args:
        mesh: The mesh on which to solve the Laplace problem.
        one_nodes: Array of node indices where the Dirichlet boundary condition is set to 1.
        zero_nodes: Array of node indices where the Dirichlet boundary condition is set to 0.

    Returns:
        Solution array corresponding to the mesh nodes.
    """
    dbc_nodes = np.concatenate((np.array(one_nodes), np.array(zero_nodes)))
    dbc_values = np.concatenate((np.ones(len(one_nodes)), np.zeros(len(zero_nodes))))

    return solve(mesh, dbc_nodes, dbc_values)


def solve(
    mesh: meshio.Mesh, boundary_nodes: np.ndarray, boundary_values: np.ndarray
) -> np.ndarray:
    """Solve a dummy Laplace-problem on a given mesh with Dirichlet boundary
    conditions."""

    K = get_global_stiffness(mesh)

    num_nod = mesh.points.shape[0]

    # apply Dirichlet boundary conditions
    free_nodes = np.setdiff1d(np.arange(num_nod), boundary_nodes)

    K_ff = K[free_nodes, :][:, free_nodes]
    K_fb = K[free_nodes, :][:, boundary_nodes]

    F_f = -K_fb.dot(boundary_values)

    # solve the system of equations
    u_f = scipy.sparse.linalg.spsolve(K_ff, F_f)

    # build the full solution vector
    u = np.zeros(num_nod)
    u[boundary_nodes] = boundary_values
    u[free_nodes] = u_f

    return u
