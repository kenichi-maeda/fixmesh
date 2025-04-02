import trimesh
import numpy as np
import open3d as o3d
import pymesh

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def detach_repair(
    input_path,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Fix self-intersections by detaching mesh.

    Parameters:
        mesh: 
            pymesh.Mesh

    Returns:
        pymesh.Mesh:
            A new mesh with no self-intersections.

    1) Load a mesh and split into submeshes.
    2) Identify smallest-volume piece as 'inner', largest-volume piece as 'outer'.
    3) Convert each to Open3D. 
    4) Iteratively shrink the inner mesh so it stays inside the outer.
    """
    tm = trimesh.load(input_path, process=False)
    submeshes = tm.split(only_watertight=True)
    
    # Sort by volume: smallesr -> "inner", larger -> "outer"
    submeshes_sorted = sorted(submeshes, key=lambda m: m.volume)
    mesh_inner_tm = submeshes_sorted[0]
    mesh_outer_tm = submeshes_sorted[-1]

    # Convert to Open3D
    mesh_inner_o3d = _trimesh_to_open3d(mesh_inner_tm)
    mesh_outer_o3d = _trimesh_to_open3d(mesh_outer_tm)

    # Compute normals for better sampling
    mesh_inner_o3d.compute_vertex_normals()
    mesh_outer_o3d.compute_vertex_normals()

    # Shrink the inner mesh
    mesh_inner_o3d = _shrink_inner_mesh(
        inner_o3d_mesh=mesh_inner_o3d,
        outer_o3d_mesh=mesh_outer_o3d,
        contact_threshold=contact_threshold,
        max_iters=max_iters,
        num_sample_points=num_sample_points
    )

    combined = mesh_inner_o3d + mesh_outer_o3d
    return combined

def _shrink_inner_mesh(
    inner_o3d_mesh: o3d.geometry.TriangleMesh,
    outer_o3d_mesh: o3d.geometry.TriangleMesh,
    contact_threshold=0.06,
    max_iters=10,
    num_sample_points=50000
):
    """
    Iteratively shrinks inner mesh so it lies inside outer mesh

    1) sample the outer mesh.
    2) see which points are inside or outside.
    3) For each iteration:
       a) Find which vertices of inner mesh are outside -> pull them inward.
       b) For vertices that are inside but too close to the outer surface,
          also pull them inward slightly.
    """

    # Build a point cloud
    pcd_outer = outer_o3d_mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_outer)
    
    # Access the inner mesh vertices
    outer_tri = _open3d_to_trimesh(outer_o3d_mesh)
    inner_verts = np.asarray(inner_o3d_mesh.vertices)
    center_inner = np.mean(inner_verts, axis=0)

    for _iter in range(max_iters):
        changed_any = False

        # Check which vertices are inside vs. outside
        inside_mask = outer_tri.contains(inner_verts)
        
        # Iterate over each vertex
        for i in range(len(inner_verts)):
            v = inner_verts[i]

            # Perform KD-tree search once
            k, idx, dist_sq = pcd_tree.search_knn_vector_3d(v, 1)
            if k == 0:
                continue  # No nearest point found, skip

            dist = np.sqrt(dist_sq[0])  # Distance to closest surface

            direction = v - center_inner
            length = np.linalg.norm(direction)
            if length > 1e-12:
                direction_unit = direction / length

            if not inside_mask[i]:
                offset = dist + contact_threshold
                if offset > 0:
                    inner_verts[i] = v - offset * direction_unit
                    changed_any = True
                continue

            # If inside, check distance to nearest surface point
            if dist < contact_threshold:
                # Pull inward so that it is at least contact_threshold from the outer surface
                offset = contact_threshold - dist
                if offset > 0:
                    inner_verts[i] = v - offset * direction_unit
                    changed_any = True

        if not changed_any:
            # No vertex changed => we've converged
            break

    inner_o3d_mesh.vertices = o3d.utility.Vector3dVector(inner_verts)
    return inner_o3d_mesh

def _trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    return o3d_mesh

def _open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)