import trimesh
import open3d as o3d
import numpy as np
import pyvista as pv
import pymesh
import matplotlib.pyplot as plt
from IPython.display import display

def voxelize(mesh_path, voxel_size=0.01):
    # Load the input mesh
    mesh = trimesh.load(mesh_path)
    components = mesh.split(only_watertight=True)

    # Assign labels
    labels = list(range(1, len(components) + 1))

    # Compute global bounds across all components
    global_origin = mesh.bounds[0]
    global_max = mesh.bounds[1]
    extent = global_max - global_origin
    grid_shape = np.ceil(extent / voxel_size).astype(int)

    # Allocate empty label array
    label_array = np.zeros(grid_shape, dtype=np.uint16)

    for i, comp in enumerate(components):
        label = labels[i]
        vgrid = comp.voxelized(voxel_size)

        # Find lower corner of voxel grid
        local_origin = vgrid.points.min(axis=0) - (voxel_size / 2)

        # Offset from global origin
        offset = ((local_origin - global_origin) / voxel_size).astype(int)

        for voxel in vgrid.sparse_indices:
            global_idx = tuple(voxel + offset)
            if all(0 <= gi < s for gi, s in zip(global_idx, grid_shape)):
                label_array[global_idx] = label

    return label_array

def visualize_label_array_with_pyvista(label_array, voxel_size=0.01):
    # Compute voxel centers and assign labels/colors
    coords = np.argwhere(label_array > 0)
    centers = (coords + 0.5) * voxel_size
    labels = label_array[tuple(coords.T)]
    
    # Create a colormap
    cmap = plt.get_cmap("tab20")
    normed = (labels - labels.min()) / (labels.max() - labels.min() + 1e-6)
    colors = (cmap(normed)[:, :3] * 255).astype(np.uint8)

    # Create PyVista point cloud (PolyData) and attach color
    point_cloud = pv.PolyData(centers)
    point_cloud["colors"] = colors 
    point_cloud["x_coord"] = point_cloud.points[:, 0]

    plotter = pv.Plotter()
    actor = plotter.add_points(
        point_cloud, scalars="colors", rgb=True,
        render_points_as_spheres=True, point_size=5
    )

    def slider_callback(value):
        mask = point_cloud["x_coord"] <= value
        clipped = point_cloud.extract_points(mask)
        actor.mapper.SetInputData(clipped)
        actor.Modified()
        plotter.render()

    x_min = point_cloud.points[:, 0].min()
    x_max = point_cloud.points[:, 0].max()

    plotter.add_slider_widget(
        callback=slider_callback,
        rng=[x_min, x_max],
        value=x_max,
        title="X Clipping",
        style='modern'
    )

    plotter.show()

def visualize_surface(mesh, min=-2, max=2):
    plotter = pv.Plotter(notebook=True)

    # Add the mesh
    mesh_actor = plotter.add_mesh(
        mesh,
        color="white",
        show_edges=True,
        edge_color="black",
        line_width=0.3
    )

    def update_clipping_plane(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh.clip_surface(new_plane, invert=False)
        mesh_actor.mapper.SetInputData(clipped_mesh)
        plotter.render()

    plotter.add_slider_widget(
        callback=update_clipping_plane,
        rng=[min, max],
        value=0,
        title="Clip Plane",
    )

    display(plotter.show(jupyter_backend="trame"))

def visualize_surface_multi(meshes, min_clip=-2, max_clip=2):
    plotter = pv.Plotter(notebook=True)

    mesh_actors = []

    for mesh in meshes:
        actor = plotter.add_mesh(
            mesh,
            color="white",
            show_edges=True,
            edge_color="black",
            line_width=0.3
        )
        mesh_actors.append((actor, mesh)) 

    def update_clipping_plane(value):
        plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        for actor, original_mesh in mesh_actors:
            clipped = original_mesh.clip_surface(plane, invert=False)
            actor.mapper.SetInputData(clipped)
        plotter.render()

    plotter.add_slider_widget(
        callback=update_clipping_plane,
        rng=[min_clip, max_clip],
        value=(min_clip + max_clip) / 2,
        title="Clip Plane (X)",
    )

    display(plotter.show(jupyter_backend="trame"))

def convert_to_pyvista(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    faces_flat = np.hstack([[3, *face] for face in faces]).astype(np.int32)
    return pv.PolyData(vertices, faces_flat)

def visualize_intersection(mesh, filename="Processed Mesh"):
    intersections = pymesh.detect_self_intersection(mesh)
    intersecting_faces = set(intersections.flatten())

    mesh = convert_to_pyvista(mesh)

    scalars = [
        1 if i in intersecting_faces else 0
        for i in range(mesh.n_faces)
    ]
    mesh.cell_data["intersections"] = scalars

    plotter = pv.Plotter(notebook=True)

    mesh_actor = plotter.add_mesh(
        mesh,
        scalars="intersections",
        color="white",
        show_edges=True,
        edge_color="black",
        line_width=0.3,
        cmap=["white", "red"],
        label=filename,
        show_scalar_bar=False
    )

    def update_clipping_plane(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh.clip_surface(new_plane, invert=False)
        mesh_actor.mapper.SetInputData(clipped_mesh)
        plotter.render()

    x_min, x_max, _, _, _, _ = mesh.bounds

    plotter.add_slider_widget(
        callback=update_clipping_plane,
        rng=[x_min, x_max],
        value=(x_min + x_max)/2,
        title="Clip Plane",
    )

    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))
