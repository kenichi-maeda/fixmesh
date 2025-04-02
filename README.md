# fixmesh

This is a repo where you can try different 3D mesh repair methods. It wraps multiple mesh processing libraries such as PyMesh, PyMeshFix, MeshLib, and more.

---

## Functions

- Self-intersection removal:
  - `pymeshfix`
  - `pymesh`
  - `meshlib`
  - `surfacenet`
  - `local_remesh`
- Mesh cutting
- Mesh detachment

---

## Installation

See `PyMesh_Installation_Guide.md` for PyMesh installation.
All requried libraries are specified in `pymesh_env.yml`.

```bash
git clone https://github.com/yourusername/fixmesh.git
cd fixmesh
