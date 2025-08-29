# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Small utilities for finite elements."""

import math

import meshio
import numpy as np
import symfem
import sympy as sp


def _create_symfem_element(cell_type: str) -> symfem.finite_element.FiniteElement:
    """Create a SymFEM element for the given cell type."""
    if cell_type == "tetra":
        return symfem.create_element("tetrahedron", "Lagrange", 1)
    elif cell_type == "tetra10":
        return symfem.create_element("tetrahedron", "Lagrange", 2)
    elif cell_type == "hexahedron":
        return symfem.create_element("hexahedron", "Lagrange", 1)

    raise ValueError(f"Unsupported cell type: {cell_type}")


# how the nodes from sysfem need to be reordered to match to the ones from meshio
_SYMFEM_NODE_REORDERING = {
    "tetra": [0, 1, 2, 3],
    "tetra10": [0, 1, 2, 3, 9, 6, 8, 7, 5, 4],
    "hexahedron": [0, 1, 3, 2, 4, 5, 7, 6],
}

# this variables stores the analytical first_derivatives of different cell types
_first_derivative_storage = {}


def ref_coords(cell_type: str) -> np.ndarray:
    """Return the reference coordinates for the nodes of a given cell type.

    Args:
        cell_type:
            The type of the cell. Supported values are:
            - "hexahedron": 8-node hexahedral element (cube).
            - "tetra": 4-node tetrahedral element.
            - "tetra10": 10-node quadratic tetrahedral element.

    Returns:
        An array of shape (n_nodes, 3) containing the reference coordinates
        of the nodes for the specified cell type.

    Raises:
        ValueError: If the provided cell_type is not supported.
    """
    element = _create_symfem_element(cell_type)
    return np.array(element.dof_plot_positions(), dtype=float)[
        _SYMFEM_NODE_REORDERING[cell_type]
    ]


def int_points_weights(
    cell_type: str, num_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute integration (quadrature) points and weights for given cell type
    and number of points.

    Args:
        cell_type:
            The type of cell for which to compute integration points and weights.
            Supported values are strings starting with "hexahedron" or "tetra".
        num_points:
            The number of integration points to use.
            Supported values:
            - For "hexahedron": 1 or 8
            - For "tetra": 1 or 4

    Returns:
        A tuple containing:
            - points: np.ndarray of shape (num_points, dim)
                The coordinates of the integration points in the reference cell.
            - weights: np.ndarray of shape (num_points,)
                The corresponding integration weights.

    Raises:
        ValueError: If the cell type or number of points is not supported.

    Examples:
    >>> points, weights = int_points_weights("hexahedron", 8)
    >>> points.shape
    (8, 3)
    >>> weights
    array([1., 1., 1., 1., 1., 1., 1., 1.])

    >>> points, weights = int_points_weights("tetra", 1)
    >>> points
    array([0.25, 0.25, 0.25])
    >>> weights
    array([0.16666667])
    """
    if cell_type.startswith("hexahedron"):
        if num_points == 1:
            return (np.array([[0.5, 0.5, 0.5]]), np.array([1.0]))
        elif num_points == 8:
            xi = 1.0 / math.sqrt(3.0)

            return (
                np.array(
                    [
                        [-1, -1, -1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, 1],
                    ]
                )
                * xi
                / 2
                + 0.5
            ), np.array([1.0 / 8] * 8)

    elif cell_type.startswith("tetra"):
        if num_points == 1:
            return (np.array([0.25, 0.25, 0.25]), np.array([1.0 / 6.0]))
        if num_points == 4:
            palpha = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
            pbeta = (5.0 - math.sqrt(5.0)) / 20.0

            return (
                np.array(
                    [
                        [pbeta, pbeta, pbeta],
                        [palpha, pbeta, pbeta],
                        [pbeta, palpha, pbeta],
                        [pbeta, pbeta, palpha],
                    ]
                ),
                np.array(
                    [1.0 / 6.0 / 4.0, 1.0 / 6.0 / 4.0, 1.0 / 6.0 / 4.0, 1.0 / 6.0 / 4.0]
                ),
            )

    raise ValueError(f"Unsupported cell type: {cell_type}")


def shape_functions(cell_type: str, xi: np.ndarray) -> np.ndarray:
    """Compute the shape functions for a given cell type at specified local
    coordinates.

    Args:
        cell_type:
            The type of finite element cell. Supported values are:
            - "hexahedron": 8-node trilinear hexahedral element
            - "tetra": 4-node linear tetrahedral element
        xi:
            The local coordinates at which to evaluate the shape functions.
            For "hexahedron", xi should be a 3-element array (xi, eta, zeta) with values in [-1, 1].
            For "tetra", xi should be a 3-element array (xi, eta, zeta) with values in [0, 1] and xi[0] + xi[1] + xi[2] <= 1.

    Returns:
        The values of the shape functions at the given local coordinates.
        - For "hexahedron": array of shape (8,)
        - For "tetra": array of shape (4,)

    Raises:
        ValueError: If the provided cell_type is not supported.
    """
    element = _create_symfem_element(cell_type)
    return element.tabulate_basis_float([xi])[_SYMFEM_NODE_REORDERING[cell_type]]


def shape_function_first_derivatives(cell_type: str, xi: np.ndarray) -> np.ndarray:
    """Compute the derivatives of shape functions with respect to the reference
    coordinates for various cell types.

    Args:
        cell_type:
            The type of finite element cell. Supported values are:
                - "hexahedron": 8-node trilinear hexahedral element
                - "tetra": 4-node linear tetrahedral element
                - "tetra10": 10-node quadratic tetrahedral element
        xi:
            The local (reference) coordinates at which to evaluate the shape function derivatives.
            For 3D elements, this should be a 1D array of length 3: [xi, eta, zeta].

    Returns:
        The derivatives of the shape functions with respect to the reference coordinates.
        The shape of the returned array is (n_nodes, 3), where n_nodes is the number of nodes
        for the specified cell type.

    Raises:
        ValueError: If an unsupported cell type is provided.
    """

    if cell_type not in _first_derivative_storage:
        element = _create_symfem_element(cell_type)
        xyz = sp.symbols("x y z")
        derivatives = [
            [sp.diff(phi, var) for var in (xyz[0], xyz[1], xyz[2])]
            for phi in element.get_basis_functions()
        ]

        _first_derivative_storage[cell_type] = sp.lambdify(xyz, derivatives, "numpy")

    return np.array(_first_derivative_storage[cell_type](*xi), dtype=float)[
        _SYMFEM_NODE_REORDERING[cell_type], :
    ]


def grad(mesh: meshio.Mesh, phi: np.ndarray) -> np.ndarray:
    """Computes the gradient of a scalar field phi defined at the nodes of the
    mesh.

    The gradient is computed at the nodes where it is averaged from the surrounding elements.

    Args:
        mesh: The input mesh.
        phi: The scalar field defined at the nodes of the mesh.

    Returns:
        The gradient of the scalar field at the nodes of the mesh.
    """

    nodal_gradients: dict[int, list[np.ndarray]] = {}

    for cell_block in mesh.cells:
        nodal_ref_coords = ref_coords(cell_block.type)

        for cell in cell_block.data:
            for node_id, ref_coord in zip(cell, nodal_ref_coords):
                if node_id not in nodal_gradients:
                    nodal_gradients[node_id] = []

                deriv = shape_function_first_derivatives(cell_block.type, ref_coord)

                J = np.matmul(deriv.T, mesh.points[cell])  # Jacobian matrix
                J_inv = np.linalg.inv(J)

                gradient = np.dot(phi[cell], np.matmul(deriv, J_inv.T))

                nodal_gradients[node_id].append(gradient)

    averaged_gradients = np.zeros((len(mesh.points), 3))

    for node_id, gradients in nodal_gradients.items():
        if gradients:
            averaged_gradients[node_id] = np.mean(gradients, axis=0)

    return averaged_gradients
