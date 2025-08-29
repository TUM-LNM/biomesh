# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Biomesh is a Python package for working with 3D meshes, providing tools for
mesh generation, manipulation, and analysis tailored for finite element
simulations of biomechanical applications."""

import pathlib
import tempfile

import lnmmeshio
import meshio

from . import laplace, mesh, run_gmsh, utils
from .adapt import lin_to_quad
from .filter import (
    filter_by_block_ids,
    filter_by_cellblock,
    filter_by_cellblock_point_mapping,
)
from .merge import merge
from .reorder import reorder


def combine_colored_stl_files(*stl_files: pathlib.Path) -> meshio.Mesh:
    """Combine multiple colored STL files into a single mesh that is
    returned."""
    return mesh.merge_colored_stl(*stl_files)[0]


def mesh_colored_stl_files(*stl_files: pathlib.Path, mesh_size: float) -> meshio.Mesh:
    """Generate a mesh from multiple colored STL files.

    Args:
        *stl_files:
            Paths to the STL files to be merged.

        mesh_size:
            The target size for the mesh elements.

    Returns:
        The generated mesh.
    """
    assert len(stl_files) > 0, "At least one STL file must be provided."

    with tempfile.TemporaryDirectory() as tmpdir:
        # merge stl files into a single mesh file
        gmsh_file = pathlib.Path(tmpdir) / "merged.mesh"
        m, surface_loops = mesh.merge_colored_stl(*stl_files)
        lnmmeshio.write_mesh(str(gmsh_file), m)

        # remesh file using Gmsh
        return run_gmsh.remesh_file(gmsh_file, surface_loops, mesh_size)
