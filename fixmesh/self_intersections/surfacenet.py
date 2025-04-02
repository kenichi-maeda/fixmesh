import pymesh
import trimesh
import numpy as np
import pyvista as pv


def fix_with_surfaceNet(mesh,  voxel_size=0.01):
    """
    Fix self-intersections using 3D SurfaceNet.

    Parameters:
        mesh: pymesh.Mesh

    Returns:
        pyvista.PolyData:
            A new mesh with labels.
    """
    mesh = _pymesh_to_trimesh(mesh)
    components = mesh.split(only_watertight=True)
    labels = list(range(1, len(components) + 1))

    # Compute global bounds from the original mesh
    global_origin = mesh.bounds[0]          # lower corner of bounding box
    global_max = mesh.bounds[1]             # upper corner of bounding box
    extent = global_max - global_origin
    grid_shape = np.ceil(extent / voxel_size).astype(int)

    # Create a global label array filled with -1 (background)
    label_array = np.zeros(grid_shape, dtype=np.uint8)

    for i, comp in enumerate(components):
        vgrid = comp.voxelized(voxel_size)

        ########
        # vgrid is a voxel grid object
        # vgrid.sparse_indices: A list of indices showing which voxels are occuied by the mesh
        # vgrid.points: The centers of these occupied voxels
        ########
        
        # Compute the local voxel grid origin from voxel centers:
        # Use the minimum point of the voxel centers and subtract half the voxel size
        local_origin = vgrid.points.min(axis=0) - (voxel_size / 2)
        
        # Compute offset (in voxels) from the global origin
        offset = ((local_origin - global_origin) / voxel_size).astype(int)
        
        # For each occupied voxel in the component, map it into the global grid
        for voxel in vgrid.sparse_indices:
            global_idx = tuple(voxel + offset)
            if all(0 <= gi < s for gi, s in zip(global_idx, grid_shape)):
                label_array[global_idx] = labels[i]

    # Create a PyVista ImageData grid.
    grid = pv.ImageData(dimensions=np.array(label_array.shape) + 1,
                        origin=global_origin,
                        spacing=(voxel_size, voxel_size, voxel_size))

    # Expand the voxel-based label array to the grid’s point data.
    point_labels = np.zeros(grid.dimensions, dtype=np.int16)
    point_labels[:-1, :-1, :-1] = label_array

    grid.point_data["labels"] = point_labels.flatten(order="F")
    grid.set_active_scalars("labels")

    contours = grid.contour_labeled(len(labels), smoothing=True, output_mesh_type='triangles')
    return contours

def _pymesh_to_trimesh(mesh):
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)