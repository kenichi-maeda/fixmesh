import pymesh

def fix_with_pymesh(mesh: 'pymesh.Mesh') -> 'pymesh.Mesh':
    """
    Fix self-intersections using PyMesh.

    Parameters:
        mesh: pymesh.Mesh

    Returns:
        pymesh.Mesh:
            A new mesh with no self-intersections.
    """
    return pymesh.compute_outer_hull(mesh)