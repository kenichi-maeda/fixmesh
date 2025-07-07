import meshlib.mrmeshpy as mrmesh
import numpy as np
import pymesh

def fix_with_meshlib(file_path, voxel_size=0.01):
    """
    Fix self-intersections using MeshLib.

    Args:
        mesh (pymesh.Mesh): The input mesh with self-intersections.
        voxel_size (float, optional): The voxel size. Defaults to 0.01.

    Returns:
        pymesh.Mesh: The repaired mesh with no self-intersections.
    """
    mesh = mrmesh.loadMesh(file_path)

    # Fix mesh
    mrmesh.fixSelfIntersections(mesh, voxel_size)

    # Revert to pymesh
    vertices = np.array([list(v) for v in mesh.points.vec]) 
    faces = []
    valid_faces = mesh.topology.getValidFaces() 
    for f in valid_faces:
        verts = mesh.topology.getTriVerts(f) 
        faces.append([verts[0].get(), verts[1].get(), verts[2].get()]) 
    faces = np.array(faces, dtype=np.int32)  

    return pymesh.form_mesh(vertices, faces)