import pymesh

def fix_with_pymesh(mesh):
    """
    Fix self-intersections using PyMesh.

    This function calls pymesh.compute_outerhull().

    Args:
        mesh (pymesh.Mesh): The input mesh with self-intersections.

    Returns:
        pymesh.Mesh: The repaired mesh with no self-intersections.
    """
    return pymesh.compute_outer_hull(mesh)