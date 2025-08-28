# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Tests for meshing stl-files."""

import biomesh
import pathlib
import meshio
import pytest
import numpy as np

_my_script_dir = pathlib.Path(__file__).parent


def test_mesh_stl():
    """Test meshing multiple stl files."""

    mesh = biomesh.mesh_colored_stl_files(
        _my_script_dir / "data" / "p1.stl",
        _my_script_dir / "data" / "p2.stl",
        _my_script_dir / "data" / "p3.stl",
        mesh_size=2.0,
    )

    assert mesh.cells_dict["line"].shape[0] == pytest.approx(482, 10)
    assert mesh.cells_dict["line"].shape[1] == 2

    assert mesh.cells_dict["triangle"].shape[0] == pytest.approx(23047, 1000)
    assert mesh.cells_dict["triangle"].shape[1] == 3

    assert mesh.cells_dict["tetra"].shape[0] == pytest.approx(77869, 1000)
    assert mesh.cells_dict["tetra"].shape[1] == 4

    assert mesh.points.shape[0] == pytest.approx(18000, 1000)
    assert mesh.points.shape[1] == 3

    assert len(mesh.cell_data["gmsh:geometrical"]) == 12

    assert all(mesh.cell_data["gmsh:geometrical"][0] == 1)
    assert len(mesh.cell_data["gmsh:geometrical"][0]) == pytest.approx(1, abs=1)

    assert all(mesh.cell_data["gmsh:geometrical"][1] == 2)
    assert len(mesh.cell_data["gmsh:geometrical"][1]) == pytest.approx(1, abs=1)

    assert all(mesh.cell_data["gmsh:geometrical"][2] == 1)
    assert len(mesh.cell_data["gmsh:geometrical"][2]) == pytest.approx(11, abs=2)

    assert all(mesh.cell_data["gmsh:geometrical"][3] == 2)
    assert len(mesh.cell_data["gmsh:geometrical"][3]) == pytest.approx(4, abs=1)

    assert all(mesh.cell_data["gmsh:geometrical"][4] == 1)
    assert len(mesh.cell_data["gmsh:geometrical"][4]) == pytest.approx(375, abs=40)

    assert all(mesh.cell_data["gmsh:geometrical"][5] == 2)
    assert len(mesh.cell_data["gmsh:geometrical"][5]) == pytest.approx(4, abs=1)

    assert all(mesh.cell_data["gmsh:geometrical"][6] == 3)
    assert len(mesh.cell_data["gmsh:geometrical"][6]) == pytest.approx(29, abs=5)

    assert all(mesh.cell_data["gmsh:geometrical"][7] == 4)
    assert len(mesh.cell_data["gmsh:geometrical"][7]) == pytest.approx(51, abs=5)

    assert all(mesh.cell_data["gmsh:geometrical"][8] == 5)
    assert len(mesh.cell_data["gmsh:geometrical"][8]) == pytest.approx(4, abs=1)

    assert all(mesh.cell_data["gmsh:geometrical"][9] == 1)
    assert len(mesh.cell_data["gmsh:geometrical"][9]) == pytest.approx(920, abs=100)

    assert all(mesh.cell_data["gmsh:geometrical"][10] == 2)
    assert len(mesh.cell_data["gmsh:geometrical"][10]) == pytest.approx(91, abs=10)

    assert all(mesh.cell_data["gmsh:geometrical"][11] == 3)
    assert len(mesh.cell_data["gmsh:geometrical"][11]) == pytest.approx(4, abs=1)
