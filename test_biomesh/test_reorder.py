import pathlib
import meshio
import biomesh
import numpy as np


def test_reorder():
    _my_script_dir = pathlib.Path(__file__).parent

    mesh = meshio.read(_my_script_dir / "data" / "test_mesh.vtu")

    # store points as data to check whether we will reorder the data correctly
    mesh.point_data["test_data"] = mesh.points

    reordered_mesh = biomesh.reorder(mesh)

    np.testing.assert_allclose(
        reordered_mesh.points[0],
        np.array([3.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        reordered_mesh.points[2],
        np.array([3.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        reordered_mesh.points[4],
        np.array([2.0, 1.0, 1.0]),
    )

    np.testing.assert_allclose(
        reordered_mesh.points[17],
        np.array([5.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        reordered_mesh.point_data["test_data"],
        reordered_mesh.points,
    )
