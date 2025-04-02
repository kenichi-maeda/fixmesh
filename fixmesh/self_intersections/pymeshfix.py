import pymesh
import pymeshfix

def fix_with_pymeshfix(mesh: 'pymesh.Mesh') -> 'pymesh.Mesh':
    """
    Fix self-intersections using PyMeshFix.

    Parameters:
        mesh: pymesh.Mesh

    Returns:
        pymesh.Mesh:
            A new mesh with no self-intersections.
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    clean_vertices, clean_faces = pymeshfix.clean_from_arrays(vertices, faces)
    return pymesh.form_mesh(clean_vertices, clean_faces)