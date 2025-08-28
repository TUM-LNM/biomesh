# biomesh

[![pipeline](https://github.com/TUM-LNM/biomesh/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/TUM-LNM/biomesh/actions/workflows/build_and_test.yml)

## Usage

To generate a finite element mesh from multiple colored stl-files, you simply need to do

```python
import biomesh

mesh = biomesh.mesh_colored_stl_files(
    "path/to/part1.stl",
    "path/to/part2.stl",
    "path/to/part3.stl",
    mesh_size=2.0
)

# make all elements quadratic
mesh = biomesh.lin_to_quad(mesh)

# reorder nodes to reduce bandwidth of matrix
mesh = biomesh.reorder(mesh)
```

The nodes of all stl-files are matched. Each stl-file will be considered as an own volume.
