# biomesh

[![pipeline](https://github.com/TUM-LNM/biomesh/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/TUM-LNM/biomesh/actions/workflows/build_and_test.yml)

## :rocket: Installation

biomesh can be installed via pip as

```
pip install biomesh
```

If you want to generate a mesh with Gmsh, you also need to install the gmsh python library as

```
pip install gmsh
```

## :book: Usage

`biomesh` is composed of multiple utilities for working with complex biomechanical geometries. Below
are some common workflows

### Generate a mesh from colored STL files

Colored STL-files (e.g., exported from Materialise 3-matic) can be used to generate a volume mesh.
Although STL color encoding is not standardized, some software packages embed surface IDs in unused
byte fields. `biomesh` leverages this to extract surface information.

:point_right: Generating a mesh from stl-files requires the `gmsh` Python package.

```python
import biomesh

mesh = biomesh.mesh_colored_stl_files(
    "path/to/part1.stl",
    "path/to/part2.stl",
    "path/to/part3.stl",
    mesh_size=2.0
)
```

Alternatively, you can load the meshes from any format supported by [meshio](https://github.com/nschloe/meshio):

```python
import meshio

meshio.read("path/to/mesh.vtu")
```

### Convert linear to quadratic elements

Convert linear elements in your mesh to quadratic ones:

```python
mesh = biomesh.lin_to_quad(mesh)
```

### Reorder mesh nodes

Finite element solvers often benefit from reducing bandwidth in the system matrix. `biomesh` provides
a node reordering algorithm based on Cuthill-McKee's algorithm to improve efficiency:

```python
mesh = biomesh.reorder(mesh)
```

### Merge multiple meshes

Combine several meshes into a single mesh object:

```python
mesh_all = biomesh.merge(mesh1, mesh2, mesh3)
```

:warning: Overlapping points are **not** automatically merged.

### Filter a mesh

Extract a subset of a mesh using flexible filters. For example, filtering by cell type:

```python
# keep only hexahedral cells
filter_hex = lambda block : block.type == 'hexahedron'
mesh_filtered = biomesh.filter.by_cellblock(mesh, filter_hex)

# Get point mapping from old -> new IDs
point_mapping = biomesh.filter.points_map_by_cellblock(mesh, filter_hex)
```

### Solve simple Laplace problem

`biomesh` can also solve basic Laplace problems, commonly used to estimate fiber directions with
rule based methods.

```python
dbc_nodes = np.array([0, 1, 2, 4])
dbc_values = np.array([0.0, 0.0, 1.0, 1.0])

phi = biomesh.solve(mesh, dbc_nodes, dbc_values)

# or equivalently
phi = biomesh.solve_onezero(mesh, np.array([2, 4]), np.array([0, 1]))
```

The result `phi` is the solution vector of the Laplace problem.

### Finite Element Utilities

`biomesh.fe` provides helper functions for finite element analysis. For example, compute the nodal
averaged gradient of a scalar field:

```python
grad_phi = biomesh.fe.grad(mesh, phi)
```
