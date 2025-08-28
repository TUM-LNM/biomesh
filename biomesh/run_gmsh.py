import pathlib
import meshio
import tempfile
from types import TracebackType, ModuleType


class GmshApi:
    """A context manager for the GMSH API."""

    def __init__(self):  # type: ignore
        try:
            import gmsh
        except ImportError:
            raise RuntimeError(
                "GMSH Python API is not available. In order to use Gmsh, you need to manually install gmsh, e.g., via pip install gmsh"
            )

        self.gmsh = gmsh

    def __enter__(self):  # type: ignore
        """Initialize gmsh api."""
        self.gmsh.initialize()
        return self.gmsh

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:  # type: ignore
        """Finalize gmsh api."""
        self.gmsh.finalize()


def remesh_file(
    file_path: pathlib.Path, surface_loops: list[set[int]], mesh_size: float
):
    with GmshApi() as gmsh:
        # read mesh file
        gmsh.merge(str(file_path))

        # create topology and geometry
        gmsh.model.mesh.createTopology()
        gmsh.model.mesh.createGeometry()

        # create surface loops
        for surface_ids in surface_loops:
            surface_loop = gmsh.model.geo.addSurfaceLoop(list(surface_ids))
            gmsh.model.geo.addVolume([surface_loop])

        # synchronize added surface loops
        gmsh.model.geo.synchronize()

        field_id = gmsh.model.mesh.field.add("MathEval", 1)
        gmsh.model.mesh.field.setString(field_id, "F", str(mesh_size))
        gmsh.model.mesh.field.setAsBackgroundMesh(field_id)

        # generate 3D mesh
        gmsh.model.mesh.generate(3)

        with tempfile.TemporaryDirectory() as dir:
            file_path = pathlib.Path(dir) / "temp.msh"
            gmsh.write(str(file_path))

            return meshio.read(str(file_path))
