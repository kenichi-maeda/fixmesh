import pymesh
import numpy as np
import os
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from numpy.linalg import norm

def fix_with_localRemesh(mesh, detail=2.4):
    """
    Fix self-intersections and remesh joint sections.

    Args:
        mesh (pymesh.Mesh): The input mesh with self-intersections.
        detail (float, optional): The value controls the finess of the remesh. Defaults to 2.4.

    Returns:
        pymesh.Mesh: The repaired mesh with no self-intersections.
    """
    intersections = pymesh.detect_self_intersection(mesh)
    intersecting_vertices, intersecting_faces = _track_self_intersecting_faces(mesh, intersections)
    outer_hull = pymesh.compute_outer_hull(mesh)
    mapped_vertices = _map_to_modified_mesh(mesh, outer_hull, intersecting_vertices)
    submesh, face_mask = _extract_self_intersecting_region_from_modified(outer_hull, mapped_vertices)
    remaining_mesh = _extract_remaining_mesh(outer_hull, face_mask)

    components = pymesh.separate_mesh(submesh)

    if len(components) == 1:
        repaired_submesh = _remesh(submesh, detail)
    else:
        repaired_components = []
        for compoenent in components:
            repaired_component = _remesh(compoenent, detail)
            repaired_components.append(repaired_component)
        repaired_submesh = pymesh.merge_meshes(repaired_components)

    aligned_submesh = _align_submesh_boundary(remaining_mesh, repaired_submesh)
    repaired_full = _replace_submesh_in_original(remaining_mesh, aligned_submesh)
    final = _refinement(repaired_full)
    return final

def _track_self_intersecting_faces(mesh, intersections):
    """
    Tracks the self-intersecting region's vertices and faces in the original mesh.
    """
    intersecting_faces = set(intersections.flatten())
    intersecting_vertices = np.unique(mesh.faces[list(intersecting_faces)].flatten())
    return intersecting_vertices, intersecting_faces

def _map_to_modified_mesh(original_mesh, modified_mesh, intersecting_vertices):
    """
    Maps the intersecting region from the original mesh to the modified mesh.

    """

    original_vertices = original_mesh.vertices
    modified_vertices = modified_mesh.vertices

    # Construct KDTree for fast nearest-neighbor search
    tree = cKDTree(modified_vertices)

    # Find the nearest modified vertex for each original intersecting vertex
    distances, nearest_indices = tree.query(original_vertices[intersecting_vertices], k=1)

    # Filter out any mappings where the distance is too large
    threshold = 1e-6
    mapped_vertices = np.array([
        nearest_indices[i] for i in range(len(intersecting_vertices)) if distances[i] < threshold
    ])

    return mapped_vertices

def _extract_self_intersecting_region_from_modified(mesh, intersecting_vertices):
    """
    Extracts the submesh corresponding to the intersecting region from the modified mesh.

    e.g.,
    From Step 3, we know the mapped intersecting vertex is [4].
    <Modified Mesh>
    Vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.51, 0.51, 0.51]]
    Faces: [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]].

    Here, we identify faces in the modified mesh that contain any of the intersecting vertices.
    For vertex 4, all four faces ([0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4])
    """
    # Step 1: Identify initial face mask
    face_mask = np.any(np.isin(mesh.faces, intersecting_vertices), axis=1)
    sub_faces = mesh.faces[face_mask]

    # Step 2: Build the initial submesh
    submesh = pymesh.form_mesh(mesh.vertices, sub_faces)

    # Step 3: Find adjacent faces
    from collections import defaultdict

    # Create an edge-to-face map for the entire mesh
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(mesh.faces):
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)

    # Collect all edges of the current submesh
    submesh_edges = set()
    for face in sub_faces:
        submesh_edges.update([
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ])

    # Find all adjacent faces to the submesh
    adjacent_faces = set()
    for edge in submesh_edges:
        for face_idx in edge_to_faces[edge]:
            if not face_mask[face_idx]:  # If the face is not already in the submesh
                adjacent_faces.add(face_idx)

    # Update the face mask to include adjacent faces
    updated_face_mask = face_mask.copy()
    updated_face_mask[list(adjacent_faces)] = True

    # Step 4: Rebuild the submesh with the updated face mask
    all_faces = mesh.faces[updated_face_mask]
    updated_submesh = pymesh.form_mesh(mesh.vertices, all_faces)

    # Step 5: Identify and remove outermost faces
    boundary_edges = defaultdict(list)
    for face_idx, face in enumerate(updated_submesh.faces):
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            boundary_edges[edge].append(face_idx)

    # Find boundary faces (faces with edges belonging to only one face)
    boundary_faces = set()
    for edge, faces in boundary_edges.items():
        if len(faces) == 1:  # Boundary edge
            boundary_faces.add(faces[0])

    # Create a mask to exclude boundary faces
    non_boundary_face_mask = np.ones(len(updated_submesh.faces), dtype=bool)
    non_boundary_face_mask[list(boundary_faces)] = False

    # Rebuild the submesh without boundary faces
    final_faces = updated_submesh.faces[non_boundary_face_mask]
    final_submesh = pymesh.form_mesh(updated_submesh.vertices, final_faces)

    # Step 6: Update the face mask to reflect the removed outermost faces
    final_face_indices = np.where(updated_face_mask)[0][non_boundary_face_mask]
    final_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    final_face_mask[final_face_indices] = True

    # Clean isolated vertices in the final submesh
    final_submesh, _ = pymesh.remove_isolated_vertices(final_submesh)

    return final_submesh, final_face_mask

def _extract_remaining_mesh(original_mesh, face_mask):
    """
    Remove the faces of the submesh from the original mesh and return the remaining part.
    """
    faces = original_mesh.faces
    keep_mask = np.logical_not(face_mask)
    kept_faces = faces[keep_mask]
    remaining_mesh = pymesh.form_mesh(original_mesh.vertices, kept_faces)
    remaining_mesh, _ = pymesh.remove_isolated_vertices(remaining_mesh)
    return remaining_mesh

def _remesh(mesh, detail=2.4e-2):

    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    target_len = diag_len * detail

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6, preserve_feature=True) # Remove extremely small edges
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        #print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.9, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh

def _compute_median_edge_length(mesh):
    """
    Compute the median edge length of a mesh. 
    """
    face_indices = mesh.faces
    verts = mesh.vertices
    
    edge_lengths = []
    for f in face_indices:
        # Each face has 3 edges: (f[0], f[1]), (f[1], f[2]), (f[2], f[0])
        e1 = np.linalg.norm(verts[f[0]] - verts[f[1]])
        e2 = np.linalg.norm(verts[f[1]] - verts[f[2]])
        e3 = np.linalg.norm(verts[f[2]] - verts[f[0]])
        edge_lengths.extend([e1, e2, e3])
    
    return np.median(edge_lengths)

def _choose_tolerance_by_median_edge(mesh, multiple=4):
    med_edge = _compute_median_edge_length(mesh)
    return multiple * med_edge

def _detect_boundary_vertices(mesh):
    """
    Detect boundary vertices in the given mesh.
    A boundary vertex is connected to at least one boundary edge.
    """
    from collections import defaultdict

    # Step 1: Build an edge-to-face mapping
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(mesh.faces):
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)

    # Step 2: Identify boundary edges (edges with only one adjacent face)
    boundary_edges = [edge for edge, faces in edge_to_faces.items() if len(faces) == 1]

    # Step 3: Collect boundary vertices
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.update(edge)

    # Step 4: Create a mask for boundary vertices
    boundary_mask = np.zeros(mesh.num_vertices, dtype=bool)
    boundary_mask[list(boundary_vertices)] = True

    return boundary_mask

def _align_submesh_boundary(remaining_mesh, repaired_submesh):
    """
    Align the boundary vertices of the repaired submesh to the original mesh's boundary.
    """
    tolerance = _choose_tolerance_by_median_edge(remaining_mesh)

    # Identify boundary vertices
    original_boundary = _detect_boundary_vertices(remaining_mesh)
    repaired_boundary = _detect_boundary_vertices(repaired_submesh)

    # Build KDTree for original mesh boundary vertices
    original_vertices = remaining_mesh.vertices
    repaired_vertices = repaired_submesh.vertices
    tree = KDTree(original_vertices[original_boundary])

    repaired_vertices = repaired_submesh.vertices.copy()
    # Snap repaired boundary vertices to original mesh
    for idx in np.where(repaired_boundary)[0]:  # Iterating over the indices of vertices marked as boundary vertices
        dist, nearest_idx = tree.query(repaired_vertices[idx], distance_upper_bound=tolerance)
        if dist < tolerance:
            # For example, suppose tolerance = 0.2
            # [1.1, 0, 0] becomes [1.0, 0, 0]
            repaired_vertices[idx] = original_vertices[original_boundary][nearest_idx]

    # Rebuild submesh with aligned vertices
    repaired_submesh = pymesh.form_mesh(repaired_vertices, repaired_submesh.faces)
    return repaired_submesh

def _replace_submesh_in_original(remaining_mesh, repaired_submesh):
    """
    Re-stitche a repaired submesh back into the original mesh.
    """
    merged = pymesh.merge_meshes([remaining_mesh, repaired_submesh])
    merged, _ = pymesh.remove_duplicated_faces(merged)
    merged, _ = pymesh.remove_isolated_vertices(merged)
    return merged

def _intermediate(mesh):
    temp_file = "temp.ply"
    pymesh.save_mesh(temp_file, mesh, ascii=True)
    mesh = pymesh.load_mesh(temp_file)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    return mesh

def _refinement(mesh):
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 175.0, 10)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    while len(pymesh.detect_self_intersection(_intermediate(mesh))) != 0:
        mesh = pymesh.resolve_self_intersection(_intermediate(mesh))
    return mesh