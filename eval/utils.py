import pymesh
import numpy as np
import pyvista as pv
from IPython.display import display
import logging
from tabulate import tabulate
import os
from scipy.spatial import KDTree
import trimesh
from scipy.spatial import cKDTree
from numpy.linalg import norm
import pymeshfix
import meshlib.mrmeshpy as mrmesh
from scipy.sparse import lil_matrix


def convert_to_pyvista(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    faces_flat = np.hstack([[3, *face] for face in faces]).astype(np.int32)
    return pv.PolyData(vertices, faces_flat)


def visualize(mesh, filename="Processed Mesh", min=-2, max=2):
    if (isinstance(mesh, pymesh.Mesh)):
        mesh = convert_to_pyvista(mesh)
    plotter = pv.Plotter(notebook=True)

    # Add the mesh
    mesh_actor = plotter.add_mesh(
        mesh,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename,
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

    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))

def visualize_two_meshes(mesh1, mesh2, filename1="Mesh 1", filename2="Mesh 2"):
    mesh1 = convert_to_pyvista(mesh1)
    mesh2 = convert_to_pyvista(mesh2)
    plotter = pv.Plotter(shape=(1, 2), notebook=True)

    plotter.subplot(0, 0)
    mesh_actor1 = plotter.add_mesh(
        mesh1,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename1,
        line_width=0.3
    )
    plotter.add_text("Mesh 1", position="upper_left", font_size=10)

    def update_clipping_plane_mesh1(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(3, 1, 0))
        clipped_mesh = mesh1.clip_surface(new_plane, invert=False)
        mesh_actor1.mapper.SetInputData(clipped_mesh)
        plotter.render()

    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh1,
        rng=[-2, 2],
        value=0,
        title="Clip Plane Mesh 1",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    plotter.subplot(0, 1)
    mesh_actor2 = plotter.add_mesh(
        mesh2,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename2,
        line_width=0.3
    )
    plotter.add_text("Mesh 2", position="upper_left", font_size=10)

    def update_clipping_plane_mesh2(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh2.clip_surface(new_plane, invert=False)
        mesh_actor2.mapper.SetInputData(clipped_mesh)
        plotter.render()

    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh2,
        rng=[-2, 2],
        value=0,
        title="Clip Plane Mesh 2",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))

def visualize_three_meshes(mesh1, mesh2, mesh3, filename1="Mesh 1", filename2="Mesh 2", filename3="Mesh 3", min=-2, max=2):
    mesh1 = convert_to_pyvista(mesh1)
    mesh2 = convert_to_pyvista(mesh2)
    mesh3 = convert_to_pyvista(mesh3)
    
    plotter = pv.Plotter(shape=(1, 3), notebook=True)

    # Mesh 1
    plotter.subplot(0, 0)
    mesh_actor1 = plotter.add_mesh(
        mesh1,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename1,
        line_width=0.3
    )
    plotter.add_text("Mesh 1", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh1(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh1.clip_surface(new_plane, invert=False)
        mesh_actor1.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh1,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 1",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    # Mesh 2
    plotter.subplot(0, 1)
    mesh_actor2 = plotter.add_mesh(
        mesh2,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename2,
        line_width=0.3
    )
    plotter.add_text("Mesh 2", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh2(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh2.clip_surface(new_plane, invert=False)
        mesh_actor2.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh2,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 2",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )
    
    # Mesh 3
    plotter.subplot(0, 2)
    mesh_actor3 = plotter.add_mesh(
        mesh3,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename3,
        line_width=0.3
    )
    plotter.add_text("Mesh 3", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh3(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh3.clip_surface(new_plane, invert=False)
        mesh_actor3.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh3,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 3",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )
    
    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))

def visualize_six_meshes(original, mesh1, mesh2, mesh3, mesh4, mesh5, min=-2, max=2):
    intersections = pymesh.detect_self_intersection(original)
    intersecting_faces = set(intersections.flatten())
    
    original = convert_to_pyvista(original)
    mesh1 = convert_to_pyvista(mesh1)
    mesh2 = convert_to_pyvista(mesh2)
    mesh3 = convert_to_pyvista(mesh3)
    #mesh4 = convert_to_pyvista(mesh4)
    mesh5 = convert_to_pyvista(mesh5)
    
    plotter = pv.Plotter(shape=(2, 3), notebook=True)

    # Original
    scalars = [
        1 if i in intersecting_faces else 0
        for i in range(original.n_faces)
    ]
    original.cell_data["intersections"] = scalars
    plotter.subplot(0, 0)
    original_actor = plotter.add_mesh(
        original,
        color="white",
        show_edges=True,
        edge_color="black",
        label="Original",
        line_width=0.3,
        scalars="intersections",
        cmap=["white", "red"], 
        show_scalar_bar=False
    )
    plotter.add_text("Original", position="upper_left", font_size=10)
    
    def update_clipping_plane_original(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = original.clip_surface(new_plane, invert=False)
        original_actor.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_original,
        rng=[min, max],
        value=0,
        title="Clip Plane Original",
    )

    # Mesh 1
    plotter.subplot(0, 1)
    mesh_actor1 = plotter.add_mesh(
        mesh1,
        color="white",
        show_edges=True,
        edge_color="black",
        label="Mesh 1",
        line_width=0.3
    )
    plotter.add_text("Mesh 1", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh1(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh1.clip_surface(new_plane, invert=False)
        mesh_actor1.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh1,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 1",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    # Mesh 2
    plotter.subplot(0, 2)
    mesh_actor2 = plotter.add_mesh(
        mesh2,
        color="white",
        show_edges=True,
        edge_color="black",
        label="Mesh 2",
        line_width=0.3
    )
    plotter.add_text("Mesh 2", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh2(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh2.clip_surface(new_plane, invert=False)
        mesh_actor2.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh2,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 2",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )
    
    # Mesh 3
    plotter.subplot(1, 0)
    mesh_actor3 = plotter.add_mesh(
        mesh3,
        color="white",
        show_edges=True,
        edge_color="black",
        label="Mesh 3",
        line_width=0.3
    )
    plotter.add_text("Mesh 3", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh3(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh3.clip_surface(new_plane, invert=False)
        mesh_actor3.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh3,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 3",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    # Mesh 4
    plotter.subplot(1, 1)
    mesh_actor4 = plotter.add_mesh(
        mesh4,
        color="white",
        show_edges=True,
        edge_color="black",
        label="Mesh 4",
        line_width=0.3
    )
    plotter.add_text("Mesh 4", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh4(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh4.clip_surface(new_plane, invert=False)
        mesh_actor4.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh4,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 4",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    # Mesh 5
    plotter.subplot(1, 2)
    mesh_actor5 = plotter.add_mesh(
        mesh3,
        color="white",
        show_edges=True,
        edge_color="black",
        label="Mesh 5",
        line_width=0.3
    )
    plotter.add_text("Mesh 5", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh5(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh5.clip_surface(new_plane, invert=False)
        mesh_actor5.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh5,
        rng=[min, max],
        value=0,
        title="Clip Plane Mesh 5",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )
    
    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))


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
        cmap=["white", "red"],  # White for normal, red for intersecting
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

def intermediate(mesh):
    temp_file = "temp.ply"
    pymesh.save_mesh(temp_file, mesh, ascii=True)
    mesh = pymesh.load_mesh(temp_file)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    return mesh

def evaluation(mesh1, mesh2):
    before_trimesh = trimesh.Trimesh(vertices=mesh1.vertices, faces=mesh1.faces)
    after_trimesh = trimesh.Trimesh(vertices=mesh2.vertices, faces=mesh2.faces)

    before_intersections = pymesh.detect_self_intersection(mesh1)
    after_intersections = pymesh.detect_self_intersection(mesh2)

    table_data = [
        ["Metric", "Before", "After"],
        ["Number of vertices", len(mesh1.vertices), len(mesh2.vertices)],
        ["Number of faces", len(mesh1.faces), len(mesh2.faces)],
        ["Number of intersecting face pairs", len(before_intersections), len(after_intersections)],
        ["Volume", before_trimesh.volume, after_trimesh.volume],
        ["Area", before_trimesh.area, after_trimesh.area],
        ["Intact vertices (%)", "Nan", _evaluate_intact_vertices(mesh1, mesh2)]
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

def full_evaluation(original, mesh1, mesh2, mesh3):
    before_trimesh = trimesh.Trimesh(vertices=original.vertices, faces=original.faces)
    after_trimesh1 = trimesh.Trimesh(vertices=mesh1.vertices, faces=mesh1.faces)
    after_trimesh2 = trimesh.Trimesh(vertices=mesh2.vertices, faces=mesh2.faces)
    after_trimesh3 = trimesh.Trimesh(vertices=mesh3.vertices, faces=mesh3.faces)

    before_pvmesh = pv.PolyData(original.vertices, np.hstack([np.full((original.faces.shape[0], 1), 3), original.faces]).astype(np.int64).flatten())
    after_pvmesh1 = pv.PolyData(mesh1.vertices, np.hstack([np.full((mesh1.faces.shape[0], 1), 3), mesh1.faces]).astype(np.int64).flatten())
    after_pvmesh2 = pv.PolyData(mesh2.vertices, np.hstack([np.full((mesh2.faces.shape[0], 1), 3), mesh2.faces]).astype(np.int64).flatten())
    after_pvmesh3 = pv.PolyData(mesh3.vertices, np.hstack([np.full((mesh3.faces.shape[0], 1), 3), mesh3.faces]).astype(np.int64).flatten())

    before_intersections = pymesh.detect_self_intersection(original)
    after_intersections1 = pymesh.detect_self_intersection(mesh1)
    after_intersections2 = pymesh.detect_self_intersection(mesh2)
    after_intersections3 = pymesh.detect_self_intersection(mesh3)

    table_data = [
        ["Metric", "Original", "Method1", "Method2", "Method3"],
        ["vertices", len(original.vertices), len(mesh1.vertices), len(mesh2.vertices), len(mesh3.vertices)],
        ["faces", len(original.faces),len(mesh1.faces), len(mesh2.faces), len(mesh3.faces)],
        ["intersecting face pairs", len(before_intersections), len(after_intersections1), len(after_intersections2), len(after_intersections3)],
        ["volume", before_trimesh.volume, after_trimesh1.volume, after_trimesh2.volume, after_trimesh3.volume],
        ["area", before_trimesh.area, after_trimesh1.area, after_trimesh2.area, after_trimesh3.area],
        ["mean displacement", "NaN", _evaluate_displacement(original, mesh1), _evaluate_displacement(original, mesh2), _evaluate_displacement(original, mesh3)],
        ["mean aspect ratio", _evaluate_aspect_ratio(before_pvmesh), _evaluate_aspect_ratio(after_pvmesh1), _evaluate_aspect_ratio(after_pvmesh2), _evaluate_aspect_ratio(after_pvmesh3)],
        ["mean condition", _evaluate_condition(before_pvmesh), _evaluate_condition(after_pvmesh1), _evaluate_condition(after_pvmesh2), _evaluate_condition(after_pvmesh3)],
        ["mean max angle", _evaluate_max_angle(before_pvmesh),  _evaluate_max_angle(after_pvmesh1), _evaluate_max_angle(after_pvmesh2), _evaluate_max_angle(after_pvmesh3)],
        ["mean min angle", _evaluate_min_angle(before_pvmesh),  _evaluate_min_angle(after_pvmesh1), _evaluate_min_angle(after_pvmesh2), _evaluate_min_angle(after_pvmesh3)],
        ["mean scaled jacobian", _evaluate_scaled_jacobian(before_pvmesh), _evaluate_scaled_jacobian(after_pvmesh1), _evaluate_scaled_jacobian(after_pvmesh2), _evaluate_scaled_jacobian(after_pvmesh3)],
        ["intact vertices (%)", "Nan", _evaluate_intact_vertices2(original, mesh1), _evaluate_intact_vertices2(original, mesh2), _evaluate_intact_vertices2(original, mesh3)],
        ["mean angle aeviation", _evaluate_angle_deviation(original), _evaluate_angle_deviation(mesh1), _evaluate_angle_deviation(mesh2), _evaluate_angle_deviation(mesh3)]
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

def full_evaluation2(original, mesh1, mesh2, mesh3, mesh4, mesh5):
    before_trimesh = trimesh.Trimesh(vertices=original.vertices, faces=original.faces)
    after_trimesh1 = trimesh.Trimesh(vertices=mesh1.vertices, faces=mesh1.faces)
    after_trimesh2 = trimesh.Trimesh(vertices=mesh2.vertices, faces=mesh2.faces)
    after_trimesh3 = trimesh.Trimesh(vertices=mesh3.vertices, faces=mesh3.faces)
    after_trimesh4 = trimesh.Trimesh(vertices=mesh4.vertices, faces=mesh4.faces)
    after_trimesh5 = trimesh.Trimesh(vertices=mesh5.vertices, faces=mesh5.faces)

    before_pvmesh = pv.PolyData(original.vertices, np.hstack([np.full((original.faces.shape[0], 1), 3), original.faces]).astype(np.int64).flatten())
    after_pvmesh1 = pv.PolyData(mesh1.vertices, np.hstack([np.full((mesh1.faces.shape[0], 1), 3), mesh1.faces]).astype(np.int64).flatten())
    after_pvmesh2 = pv.PolyData(mesh2.vertices, np.hstack([np.full((mesh2.faces.shape[0], 1), 3), mesh2.faces]).astype(np.int64).flatten())
    after_pvmesh3 = pv.PolyData(mesh3.vertices, np.hstack([np.full((mesh3.faces.shape[0], 1), 3), mesh3.faces]).astype(np.int64).flatten())
    after_pvmesh4 = pv.PolyData(mesh4.vertices, np.hstack([np.full((mesh4.faces.shape[0], 1), 3), mesh4.faces]).astype(np.int64).flatten())
    after_pvmesh5 = pv.PolyData(mesh5.vertices, np.hstack([np.full((mesh5.faces.shape[0], 1), 3), mesh5.faces]).astype(np.int64).flatten())

    before_intersections = pymesh.detect_self_intersection(original)
    after_intersections1 = pymesh.detect_self_intersection(mesh1)
    after_intersections2 = pymesh.detect_self_intersection(mesh2)
    after_intersections3 = pymesh.detect_self_intersection(mesh3)
    after_intersections4 = pymesh.detect_self_intersection(mesh4)
    after_intersections5 = pymesh.detect_self_intersection(mesh5)

    table_data = [
        ["Metric", "Original", "PyMeshFix", "MeshFix", "Meshlib", "SurfaceNets", "Local Remesh"],
        ["vertices", len(original.vertices), len(mesh1.vertices), len(mesh2.vertices), len(mesh3.vertices), len(mesh4.vertices), len(mesh5.vertices)],
        ["faces", len(original.faces),len(mesh1.faces), len(mesh2.faces), len(mesh3.faces), len(mesh4.faces), len(mesh5.faces)],
        ["intersecting face pairs", len(before_intersections), len(after_intersections1), len(after_intersections2), len(after_intersections3), len(after_intersections4), len(after_intersections5)],
        ["volume", before_trimesh.volume, after_trimesh1.volume, after_trimesh2.volume, after_trimesh3.volume, after_trimesh4.volume, after_trimesh5.volume],
        ["area", before_trimesh.area, after_trimesh1.area, after_trimesh2.area, after_trimesh3.area, after_trimesh4.area, after_trimesh5.area],
        ["mean aspect ratio", _evaluate_aspect_ratio(before_pvmesh), _evaluate_aspect_ratio(after_pvmesh1), _evaluate_aspect_ratio(after_pvmesh2), _evaluate_aspect_ratio(after_pvmesh3), _evaluate_aspect_ratio(after_pvmesh4), _evaluate_aspect_ratio(after_pvmesh5)],
        ["mMean condition", _evaluate_condition(before_pvmesh), _evaluate_condition(after_pvmesh1), _evaluate_condition(after_pvmesh2), _evaluate_condition(after_pvmesh3), _evaluate_condition(after_pvmesh4), _evaluate_condition(after_pvmesh5)],
        ["mean max angle", _evaluate_max_angle(before_pvmesh),  _evaluate_max_angle(after_pvmesh1), _evaluate_max_angle(after_pvmesh2), _evaluate_max_angle(after_pvmesh3), _evaluate_max_angle(after_pvmesh4), _evaluate_max_angle(after_pvmesh5)],
        ["mean min angle", _evaluate_min_angle(before_pvmesh),  _evaluate_min_angle(after_pvmesh1), _evaluate_min_angle(after_pvmesh2), _evaluate_min_angle(after_pvmesh3), _evaluate_min_angle(after_pvmesh4), _evaluate_min_angle(after_pvmesh5)],
        ["mean scaled jacobian", _evaluate_scaled_jacobian(before_pvmesh), _evaluate_scaled_jacobian(after_pvmesh1), _evaluate_scaled_jacobian(after_pvmesh2), _evaluate_scaled_jacobian(after_pvmesh3), _evaluate_scaled_jacobian(after_pvmesh4), _evaluate_scaled_jacobian(after_pvmesh5)],
        ["mean displacement", "NaN", _evaluate_displacement(original, mesh1), _evaluate_displacement(original, mesh2), _evaluate_displacement(original, mesh3), _evaluate_displacement(original, mesh4), _evaluate_displacement(original, mesh5)],
        ["mean angle deviation", _evaluate_angle_deviation(original), _evaluate_angle_deviation(mesh1), _evaluate_angle_deviation(mesh2), _evaluate_angle_deviation(mesh3), _evaluate_angle_deviation(mesh4), _evaluate_angle_deviation(mesh5)],
        ["intact vertices (%)", "Nan", _evaluate_intact_vertices2(original, mesh1), _evaluate_intact_vertices2(original, mesh2), _evaluate_intact_vertices2(original, mesh3), _evaluate_intact_vertices2(original, mesh4), _evaluate_intact_vertices2(original, mesh5)]
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

def _evaluate_aspect_ratio(pvmesh):
    qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
    quality_array = qual['CellQuality']
    return quality_array.mean()

def _evaluate_condition(pvmesh):
    qual = pvmesh.compute_cell_quality(quality_measure='condition')
    quality_array = qual['CellQuality']
    return quality_array.mean()

def _evaluate_max_angle(pvmesh):
    qual = pvmesh.compute_cell_quality(quality_measure='max_angle')
    quality_array = qual['CellQuality']
    return quality_array.mean()

def _evaluate_min_angle(pvmesh):
    qual = pvmesh.compute_cell_quality(quality_measure='min_angle')
    quality_array = qual['CellQuality']
    return quality_array.mean()

def _evaluate_scaled_jacobian(pvmesh):
    qual = pvmesh.compute_cell_quality(quality_measure='scaled_jacobian')
    quality_array = qual['CellQuality']
    return quality_array.mean()

def _evaluate_intact_vertices(original_mesh, repaired_mesh):
    original_vertices = original_mesh.vertices
    repaired_vertices = repaired_mesh.vertices

    original_set = set(map(tuple, original_vertices))
    repaired_set = set(map(tuple, repaired_vertices))

    intact_count = len(original_set.intersection(repaired_set))
    intact_percentage = (intact_count / len(original_set)) * 100

    return intact_percentage

def _evaluate_intact_vertices2(original_mesh, repaired_mesh, decimals=4):
    original_rounded = np.round(original_mesh.vertices, decimals=decimals)
    repaired_rounded = np.round(repaired_mesh.vertices, decimals=decimals)

    original_set = set(map(tuple, original_rounded))
    repaired_set = set(map(tuple, repaired_rounded))

    intact_count = len(original_set.intersection(repaired_set))

    intact_percentage = (intact_count / len(original_set)) * 100.0
    return intact_percentage


def _evaluate_angle_deviation(mesh):

    vertices = mesh.vertices
    faces = mesh.faces
    
    deviations = []

    for face in faces:
        tri_pts = vertices[face]

        # Compute angles at each vertex of the triangle
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3

            # Vector from current vertex to the other two vertices
            v1 = tri_pts[j] - tri_pts[i]
            v2 = tri_pts[k] - tri_pts[i]

            denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-14)
            cos_angle = np.dot(v1, v2) / denom

            # Clamp to avoid floating-precision issues outside [-1, 1]
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            angle_degrees = np.degrees(np.arccos(cos_angle))
            deviations.append(abs(angle_degrees - 60.0))

    if len(deviations) == 0:
        return 0.0

    return np.mean(deviations)


def extract_submesh_by_bbox(mesh, local_min, local_max):
    """
    Extracts a submesh from the input mesh that includes only the faces whose
    vertices lie within the specified axis-aligned bounding box [local_min, local_max].

    Parameters:
        mesh: The input mesh object.
        local_min: A 1D array defining the minimum (x, y, z) coordinates of the bounding box.
        local_max: A 1D array defining the maximum (x, y, z) coordinates of the bounding box.

    Returns:
        sub_mesh: The extracted submesh.
        face_mask: A boolean array indicating which faces of the original mesh were included in the submesh.
    """
    verts = mesh.vertices
    # inside_mask[i] is True if all x, y, and z in verts[i] lie within [local_min, local_max].
    inside_mask = np.all((verts >= local_min) & (verts <= local_max), axis=1)

    faces = mesh.faces
    # face_mask[i] is True if at least one of the three vertices of faces[i] lies within [local_min, local_max]
    face_mask = inside_mask[faces].any(axis=1)

    # Extract the subset of faces that satisfy the condition.
    sub_faces = faces[face_mask]

    # Create a submesh.
    sub_mesh = pymesh.form_mesh(verts, sub_faces)

    # Remove isolated vertices to ensure a clean submesh.
    sub_mesh, _ = pymesh.remove_isolated_vertices(sub_mesh)
    
    return sub_mesh, face_mask

def extract_remaining_mesh(original_mesh, face_mask):
    """
    Remove the faces of the submesh from the original mesh and return the remaining part.
    """
    faces = original_mesh.faces
    keep_mask = np.logical_not(face_mask)
    kept_faces = faces[keep_mask]
    remaining_mesh = pymesh.form_mesh(original_mesh.vertices, kept_faces)
    remaining_mesh, _ = pymesh.remove_isolated_vertices(remaining_mesh)
    return remaining_mesh

def detect_boundary_vertices(mesh):
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


def align_submesh_boundary(remaining_mesh, repaired_submesh, tolerance=3):
    """
    Align the boundary vertices of the repaired submesh to the original mesh's boundary.
    """

    # Identify boundary vertices
    original_boundary = detect_boundary_vertices(remaining_mesh)
    repaired_boundary = detect_boundary_vertices(repaired_submesh)

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


def replace_submesh_in_original(remaining_mesh, repaired_submesh):
    """
    Re-stitche a repaired submesh back into the original mesh.

    Steps:
    1. Remove the old faces from the original (the region we're replacing).
    2. Combine the leftover portion of the original with the newly repaired submesh
       using boolean union or just form_mesh + remove_duplicates.
    """
    merged = pymesh.merge_meshes([remaining_mesh, repaired_submesh])
    merged, _ = pymesh.remove_duplicated_faces(merged)
    merged, _ = pymesh.remove_isolated_vertices(merged)
    return merged

def _evaluate_displacement(mesh1, mesh2):

    original_vertices = mesh1.vertices
    repaired_vertices = mesh2.vertices

    # Build KD-Trees for efficient nearest-neighbor search
    original_tree = cKDTree(original_vertices)
    repaired_tree = cKDTree(repaired_vertices)

    # For each vertex in the original mesh, the distance to the nearest vertex in the repaired mesh.
    distances_original_to_repaired, _ = original_tree.query(repaired_vertices)

    # For each vertex in the repaired mesh, the distance to the nearest vertex in the original mesh.
    distances_repaired_to_original, _ = repaired_tree.query(original_vertices)

    # Combine distances for bidirectional matching
    # Why?
    # - Some vertices in the original mesh may not have a corresponding counterpart in the repaired mesh (e.g., due to merging or removal).
    # - Similarly, new vertices in the repaired mesh may not have counterparts in the original mesh (e.g., due to splitting or addition).
    all_distances = np.concatenate([distances_original_to_repaired, distances_repaired_to_original])

    max_displacement = np.max(all_distances)
    mean_displacement = np.mean(all_distances)

    return mean_displacement


def evaluate_displacement(mesh1, mesh2):

    original_vertices = mesh1.vertices
    repaired_vertices = mesh2.vertices

    # Build KD-Trees for efficient nearest-neighbor search
    original_tree = cKDTree(original_vertices)
    repaired_tree = cKDTree(repaired_vertices)

    # For each vertex in the original mesh, the distance to the nearest vertex in the repaired mesh.
    distances_original_to_repaired, _ = original_tree.query(repaired_vertices)

    # For each vertex in the repaired mesh, the distance to the nearest vertex in the original mesh.
    distances_repaired_to_original, _ = repaired_tree.query(original_vertices)

    # Combine distances for bidirectional matching
    # Why?
    # - Some vertices in the original mesh may not have a corresponding counterpart in the repaired mesh (e.g., due to merging or removal).
    # - Similarly, new vertices in the repaired mesh may not have counterparts in the original mesh (e.g., due to splitting or addition).
    all_distances = np.concatenate([distances_original_to_repaired, distances_repaired_to_original])

    max_displacement = np.max(all_distances)
    mean_displacement = np.mean(all_distances)

    print(f"Max Vertex Displacement: {max_displacement}")
    print(f"Mean Vertex Displacement: {mean_displacement}")

def track_self_intersecting_faces(mesh, intersections):
    """
    Tracks the self-intersecting region's vertices and faces in the original mesh.
    """
    intersecting_faces = set(intersections.flatten())
    intersecting_vertices = np.unique(mesh.faces[list(intersecting_faces)].flatten())
    return intersecting_vertices, intersecting_faces

def map_to_modified_mesh(original_mesh, modified_mesh, intersecting_vertices):
    """
    Maps the intersecting region from the original mesh to the modified mesh.


    e.g.,
    <Original Mesh>
    Vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5]]
    Faces: [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    Intersecting vertices (detected): [4].

    <Modified Mesh>
    Vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.51, 0.51, 0.51]]

    The original vertex [0.5, 0.5, 0.5] is closest to [0.51, 0.51, 0.51] in the modified mesh.
    So, mapped vertex in modified_mesh is [4]
    """
    # Build a mapping from original vertices to modified vertices
    original_to_modified = {}
    for i, vertex in enumerate(original_mesh.vertices):
        distances = np.linalg.norm(modified_mesh.vertices - vertex, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 1e-6:
            original_to_modified[i] = closest_idx

    # Map intersecting vertices to the modified mesh
    mapped_vertices = [original_to_modified[v] for v in intersecting_vertices if v in original_to_modified]
    return np.array(mapped_vertices)

def map_to_modified_mesh2(original_mesh, modified_mesh, intersecting_vertices):
    """
    Optimized version of mapping the intersecting region from the original mesh to the modified mesh.

    """

    original_vertices = original_mesh.vertices
    modified_vertices = modified_mesh.vertices

    # Construct KDTree for fast nearest-neighbor search
    tree = cKDTree(modified_vertices)

    # Find the nearest modified vertex for each original intersecting vertex
    distances, nearest_indices = tree.query(original_vertices[intersecting_vertices], k=1)

    # Filter out any mappings where the distance is too large (optional thresholding)
    threshold = 1e-6
    mapped_vertices = np.array([
        nearest_indices[i] for i in range(len(intersecting_vertices)) if distances[i] < threshold
    ])

    return mapped_vertices

def extract_self_intersecting_region_from_modified(mesh, intersecting_vertices):
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

def refinement(mesh):
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 175.0, 10)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    while len(pymesh.detect_self_intersection(intermediate(mesh))) != 0:
        mesh = pymesh.resolve_self_intersection(intermediate(mesh))
    return mesh

def iterative_repair(mesh, with_rounding=True, precision=11, max_iterations=15):
    """
    Resolves self-intersections in a mesh iteratively.

    Parameters:
        mesh: The input mesh object.
        with_rounding (bool): Enables rounding of vertices for stability. Default is True.
        precision (int): Rounding precision level for vertices.
        max_iterations (int): Maximum number of iterations allowed to resolve intersections.

    Returns:
        mesh: The processed mesh with no self-intersections.
    """
    # Initial rounding of vertices
    if (with_rounding):
        mesh = pymesh.form_mesh(
                np.round(mesh.vertices, precision),
                mesh.faces);
    intersecting_faces = pymesh.detect_self_intersection(mesh);

    # Iterative process to resolve self-intersections
    counter = 0;
    while len(intersecting_faces) > 0 and counter < max_iterations:
        if (with_rounding):
            involved_vertices = np.unique(mesh.faces[intersecting_faces].ravel());

            # Round only the involved vertices
            # Suppose precision = 4. Then,
            # [1.234567, 2.345678, 3.456789] <- One vertex example (x, y, z coords)
            # becomes
            # [1.23, 2.35, 3.46]
            vertices_copy = mesh.vertices.copy()  
            vertices_copy[involved_vertices, :] =\
                    np.round(mesh.vertices[involved_vertices, :],
                            precision//2);
        
            mesh = pymesh.form_mesh(vertices_copy, mesh.faces) 

        mesh = pymesh.resolve_self_intersection(mesh, "igl");
        mesh, __ = pymesh.remove_duplicated_faces(mesh, fins_only=True);
        if (with_rounding):
            mesh = pymesh.form_mesh(
                    np.round(mesh.vertices, precision),
                    mesh.faces);
        mesh = intermediate(mesh) # Reload mesh. Otherwise, the next step fails in some cases.
        intersecting_faces = pymesh.detect_self_intersection(mesh);
        print(len(intersecting_faces))
        counter += 1;

    if len(intersecting_faces) > 0:
        logging.warn("Resolving failed: max iteration reached!");

    return mesh

def remesh(mesh, detail=2.4e-2):

    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    target_len = diag_len * detail
    #print("Target resolution: {} mm".format(target_len))

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

def direct_repair(mesh):
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    clean_vertices, clean_faces = pymeshfix.clean_from_arrays(vertices, faces)
    return pymesh.form_mesh(clean_vertices, clean_faces)

def repair_with_local_remesh(mesh):
    intersections = pymesh.detect_self_intersection(mesh)
    intersecting_vertices, intersecting_faces = track_self_intersecting_faces(mesh, intersections)
    outer_hull = pymesh.compute_outer_hull(mesh)
    mapped_vertices = map_to_modified_mesh2(mesh, outer_hull, intersecting_vertices)
    submesh, face_mask = extract_self_intersecting_region_from_modified(outer_hull, mapped_vertices)
    remaining_mesh = extract_remaining_mesh(outer_hull, face_mask)

    components = pymesh.separate_mesh(submesh)

    if len(components) == 1:
        repaired_submesh = remesh(submesh)
    else:
        repaired_components = []
        for compoenent in components:
            repaired_component = remesh(compoenent)
            repaired_components.append(repaired_component)
        repaired_submesh = pymesh.merge_meshes(repaired_components)

    aligned_submesh = align_submesh_boundary2(remaining_mesh, repaired_submesh)
    repaired_full = replace_submesh_in_original(remaining_mesh, aligned_submesh)
    final = refinement(repaired_full)
    return final


def compute_average_edge_length(mesh):
    """Compute the average edge length of a mesh by extracting unique edges."""
    faces = mesh.faces  # Get face connectivity
    vertices = mesh.vertices  # Get vertex coordinates

    # Extract all edges (each face contributes 3 edges)
    edges = np.vstack([
        faces[:, [0, 1]],  # Edge between vertex 0 and 1
        faces[:, [1, 2]],  # Edge between vertex 1 and 2
        faces[:, [2, 0]]   # Edge between vertex 2 and 0
    ])

    # Remove duplicate edges
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    # Compute edge lengths
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    return np.mean(edge_lengths)

def repair_meshlib(path, details=0.08):
    mesh = mrmesh.loadMesh(path)

    # Fix mesh
    mrmesh.fixSelfIntersections(mesh, details)

    # Revert to pymesh
    vertices = np.array([list(v) for v in mesh.points.vec]) 
    faces = []
    valid_faces = mesh.topology.getValidFaces() 
    for f in valid_faces:
        verts = mesh.topology.getTriVerts(f) 
        faces.append([verts[0].get(), verts[1].get(), verts[2].get()]) 
    faces = np.array(faces, dtype=np.int32)  

    return pymesh.form_mesh(vertices, faces)

"""
def repair_contour(mesh,  voxel_size=0.01):
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Voxelization
    voxelized = mesh.voxelized(voxel_size)
    # Convert to a NumPy array
    voxel_indices = voxelized.sparse_indices
    grid_shape = voxelized.shape
    voxel_mesh = voxelized.as_boxes()  # Convert voxels to a mesh of small cubes

    label_array = np.zeros(grid_shape, dtype=np.uint16)

    # Fill the voxel grid with label '1'
    for voxel in voxel_indices:
        label_array[tuple(voxel)] = 1

    grid = pv.ImageData(dimensions=label_array.shape)
    grid.spacing = (voxel_size, voxel_size, voxel_size)  # Set voxel spacing

    # Attach label data to the grid
    grid.point_data["labels"] = label_array.flatten(order="F")  

    # Extract mesh using contour_labeled
    contours = grid.contour_labeled(smoothing=True, output_mesh_type='triangles')

    # Revert to pymesh
    vertices = contours.points 
    faces = contours.faces.reshape(-1, 4)[:, 1:]

    return pymesh.form_mesh(vertices, faces)
"""

def pymesh_to_trimesh(mesh):
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

def pyvista_to_pymesh(pv_mesh):
    vertices = pv_mesh.points 
    faces_flat = pv_mesh.faces.reshape((-1, 4))
    faces = faces_flat[:, 1:]
    pymesh_mesh = pymesh.form_mesh(vertices, faces)
    return pymesh_mesh

def repair_contour_original(mesh_path):
    mesh = pymesh.load_mesh(mesh_path)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    return mesh

def repair_contour(mesh,  voxel_size=0.01):
    mesh = pymesh_to_trimesh(mesh)
    components = mesh.split(only_watertight=True)
    labels = list(range(1, len(components) + 1))

    # Compute global bounds from the original mesh
    global_origin = mesh.bounds[0]          # lower corner of bounding box
    global_max = mesh.bounds[1]             # upper corner of bounding box
    extent = global_max - global_origin
    grid_shape = np.ceil(extent / voxel_size).astype(int)

    # Create a global label array filled with -1 (background)
    #label_array = np.full(grid_shape, -1, dtype=np.int16)
    #label_array = lil_matrix(grid_shape, dtype=np.uint8)
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


def compute_median_edge_length(mesh):
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

def choose_tolerance_by_median_edge(mesh, multiple=4):
    """
    Choose a snapping tolerance as some multiple of the median edge length.
    """
    med_edge = compute_median_edge_length(mesh)
    return multiple * med_edge

def align_submesh_boundary2(remaining_mesh, repaired_submesh):
    """
    Align the boundary vertices of the repaired submesh to the original mesh's boundary.
    """
    tolerance = choose_tolerance_by_median_edge(remaining_mesh)

    # Identify boundary vertices
    original_boundary = detect_boundary_vertices(remaining_mesh)
    repaired_boundary = detect_boundary_vertices(repaired_submesh)

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

def vis_for_paper(
    original, mesh1, mesh2, mesh3, mesh4, mesh5
):
    # CAUTION: Mesh4 is pyvista
    intersections = pymesh.detect_self_intersection(original)
    intersecting_faces = set(intersections.flatten())

    original_pv = convert_to_pyvista(original)
    mesh1_pv = convert_to_pyvista(mesh1)
    mesh2_pv = convert_to_pyvista(mesh2)
    mesh3_pv = convert_to_pyvista(mesh3)
    #mesh4_pv = convert_to_pyvista(mesh4)
    mesh5_pv = convert_to_pyvista(mesh5)

    scalars = [
        1 if i in intersecting_faces else 0
        for i in range(original_pv.n_faces)
    ]
    original_pv.cell_data["intersections"] = scalars

    plotter = pv.Plotter(shape=(2, 3), notebook=True, border=False)
    
    # Subplot (0, 0): Original
    plotter.subplot(0, 0)
    plotter.add_mesh(
        original_pv,
        show_edges=True,
        edge_color='black',
        line_width=0.3,
        label='Original',
        scalars="intersections",
        cmap=["white", "red"],
        show_scalar_bar=False
    )
    plotter.add_text('Original', position='upper_left', font_size=15)

    # Subplot (0, 1): Mesh 1
    plotter.subplot(0, 1)
    plotter.add_mesh(
        mesh1_pv, color='white', show_edges=True, edge_color='black',
        line_width=0.3, label='Mesh 1'
    )
    plotter.add_text('PyMeshFix', position='upper_left', font_size=15)

    # Subplot (0, 2): Mesh 2
    plotter.subplot(0, 2)
    plotter.add_mesh(
        mesh2_pv, color='white', show_edges=True, edge_color='black',
        line_width=0.3, label='Mesh 2'
    )
    plotter.add_text('PyMesh', position='upper_left', font_size=15)

    # Subplot (1, 0): Mesh 3
    plotter.subplot(1, 0)
    plotter.add_mesh(
        mesh3_pv, color='white', show_edges=True, edge_color='black',
        line_width=0.3, label='Mesh 3'
    )
    plotter.add_text('MeshLib', position='upper_left', font_size=15)

    # Subplot (1, 1): Mesh 4
    plotter.subplot(1, 1)
    plotter.add_mesh(
        mesh4, show_edges=True, edge_color='black',
        line_width=0.3, label='Mesh 4', scalars="BoundaryLabels", cmap="viridis", show_scalar_bar=False
    )
    plotter.add_text('SurfaceNets', position='upper_left', font_size=15)

    # Subplot (1, 2): Mesh 5
    plotter.subplot(1, 2)
    plotter.add_mesh(
        mesh5_pv, color='white', show_edges=True, edge_color='black',
        line_width=0.3, label='Mesh 5'
    )
    plotter.add_text('Local Remesh', position='upper_left', font_size=15)

    plotter.link_views()
    display(plotter.show(jupyter_backend='trame'))


def vis_for_paper_inner(
    original, mesh1, mesh2, mesh3, mesh4, mesh5
):

    # CAUTION: Mesh4 is pyvista
    intersections = pymesh.detect_self_intersection(original)
    intersecting_faces = set(intersections.flatten())

    original_pv = convert_to_pyvista(original)
    mesh1_pv = convert_to_pyvista(mesh1)
    mesh2_pv = convert_to_pyvista(mesh2)
    mesh3_pv = convert_to_pyvista(mesh3)
    mesh4_pv = mesh4
    mesh5_pv = convert_to_pyvista(mesh5)

    scalars = [
        1 if i in intersecting_faces else 0
        for i in range(original_pv.n_faces)
    ]
    original_pv.cell_data["intersections"] = scalars

    meshes_pv = [original_pv, mesh1_pv, mesh2_pv, mesh3_pv, mesh4_pv, mesh5_pv]
    plotter = pv.Plotter(shape=(2, 3), notebook=True, border=False)

    mesh_actors = []

    plotter.subplot(0, 0)
    actor0 = plotter.add_mesh(
        original_pv,
        show_edges=True,
        edge_color='black',
        line_width=0.3,
        label='Original',
        scalars="intersections",
        cmap=["white", "red"],
        show_scalar_bar=False)
    mesh_actors.append(actor0)
    plotter.add_text("Original", position='upper_left', font_size=15)

    plotter.subplot(0, 1)
    actor1 = plotter.add_mesh(mesh1_pv, color='white', show_edges=True, edge_color='black')
    mesh_actors.append(actor1)
    plotter.add_text("PyMeshFix", position='upper_left', font_size=15)

    plotter.subplot(0, 2)
    actor2 = plotter.add_mesh(mesh2_pv, color='white', show_edges=True, edge_color='black')
    mesh_actors.append(actor2)
    plotter.add_text("PyMesh", position='upper_left', font_size=15)

    plotter.subplot(1, 0)
    actor3 = plotter.add_mesh(mesh3_pv, color='white', show_edges=True, edge_color='black')
    mesh_actors.append(actor3)
    plotter.add_text("MeshLib", position='upper_left', font_size=15)

    plotter.subplot(1, 1)
    actor4 = plotter.add_mesh(
        mesh4_pv, show_edges=True, edge_color='black', scalars="BoundaryLabels", cmap="viridis",
        show_scalar_bar=False
    )
    mesh_actors.append(actor4)
    plotter.add_text("SurfaceNets", position='upper_left', font_size=15)

    plotter.subplot(1, 2)
    actor5 = plotter.add_mesh(mesh5_pv, color='white', show_edges=True, edge_color='black')
    mesh_actors.append(actor5)
    plotter.add_text("Local Remesh", position='upper_left', font_size=15)

    plotter.link_views()

    def update_clipping_plane(value):
        plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        for i, mesh_pv in enumerate(meshes_pv):
            clipped = mesh_pv.clip_surface(plane, invert=False)
            mesh_actors[i].mapper.SetInputData(clipped)
        plotter.render()

    x_min, x_max, _, _, _, _ = original_pv.bounds
    plotter.add_slider_widget(
        callback=update_clipping_plane,
        rng=[x_min, x_max],
        value=(x_min + x_max) * 0.5,
        title="Clip Plane (X)"
    )

    display(plotter.show(jupyter_backend='trame'))
    return plotter
