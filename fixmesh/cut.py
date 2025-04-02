import trimesh
import numpy as np
import pymesh


def cut_repair(mesh):
    """
    Fix self-intersections by cutting mesh.

    Parameters:
        mesh: 
            pymesh.Mesh

    Returns:
        pymesh.Mesh:
            A new mesh with no self-intersections.
    """

    count = _count_num_components(mesh)
    intersecting_faces = pymesh.detect_self_intersection(mesh)
    intersected_faces = set(intersecting_faces.flatten())

    remaining_faces = np.setdiff1d(np.arange(mesh.num_faces), list(intersected_faces))
    remaining_faces_vertices = mesh.faces[remaining_faces]

    unique_vertices, new_faces = np.unique(remaining_faces_vertices, return_inverse=True)
    new_mesh = pymesh.form_mesh(mesh.vertices[unique_vertices], new_faces.reshape(-1, 3))

    submeshes = _pymesh_to_trimesh(new_mesh).split(only_watertight=False)
    submeshes_sorted = sorted(submeshes, key=lambda x: len(x.vertices), reverse=True)
    submeshes_needed = submeshes_sorted[:count]

    repaired_submesh_needed = []
    for submesh in submeshes_needed:
        submesh = pymesh.convex_hull(_trimesh_to_pymesh(submesh))
        submesh = _collapse_long_edges(submesh)
        submesh = _pymesh_to_trimesh(submesh)
        repaired_submesh_needed.append(submesh)

    final = trimesh.util.concatenate(repaired_submesh_needed)
    return final

def _pymesh_to_trimesh(mesh):
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

def _trimesh_to_pymesh(mesh):
    return pymesh.form_mesh(vertices=mesh.vertices, faces=mesh.faces)

def _compute_median_edge_length(mesh):
    face_indices = mesh.faces 
    verts = mesh.vertices

    edge_lengths = []
    unique_edges = set()


    for f in face_indices:
        # Each face has 3 edges: (f[0], f[1]), (f[1], f[2]), (f[2], f[0])
        edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        
        for edge in edges:
            edge = tuple(sorted(edge))
            if edge not in unique_edges:
                unique_edges.add(edge)
                
                # Calculate the edge length
                e_length = np.linalg.norm(verts[edge[0]] - verts[edge[1]])
                edge_lengths.append(e_length)

    return np.median(edge_lengths) if edge_lengths else 0

def _collapse_long_edges(mesh):
    tol = _compute_median_edge_length(mesh) * 1.6
    new_mesh, _ = pymesh.split_long_edges(mesh, tol)
    return new_mesh

def _count_num_components(mesh):
    mesh = _pymesh_to_trimesh(mesh)
    components = mesh.split(only_watertight=True)
    return len(components)