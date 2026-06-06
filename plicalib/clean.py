import numpy as np
import open3d as o3d
from typing import Tuple, Optional, List
import os
import contextlib
import warnings
from plicalib.meshes import compute_face_areas

def remove_small_triangles(mesh, area_min):
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    areas = compute_face_areas(V, T)
    remove_mask = areas < area_min

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    return mesh


@contextlib.contextmanager
def _suppress_c_output():
    devnull = os.open(os.devnull, os.O_WRONLY)

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)

        os.close(old_stdout)
        os.close(old_stderr)

def clean_mesh_open3d(mesh : o3d.geometry.TriangleMesh,
                    taubin_smoothing_iterations : int = 10,
                    merge_vertices_relative_scale : Optional[float] = None,
                    hole_filling_num_neighbors : Optional[int] = None):
    
    '''
        The mesh is processed as follows:
        1. Remove duplicated triangles, duplicated vertices, degenerate triangles, non-manifold edges
        2. Keep only the largest connected component
        3. Optionally merge close vertices (if merge_vertices_relative_scale is not None and > 0)
        4. Optionally fill holes (if hole_filling_num_neighbors is not None and > 0). This uses pymeshfix, which is a wrapper around the MeshFix library. If pymeshfix is not installed, this step is skipped and a warning is issued.
        5. Optionally apply Taubin smoothing (if taubin_smoothing_iterations > 0)
    '''
    
    # default repair/clean operations
    c_mesh = mesh.remove_duplicated_triangles()
    c_mesh = c_mesh.remove_duplicated_vertices()
    c_mesh = c_mesh.remove_degenerate_triangles()
    c_mesh = c_mesh.remove_non_manifold_edges()
    c_mesh = c_mesh.remove_unreferenced_vertices()

    # Keep only the largest connected component
    clusters, lengths, _ = c_mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    lengths = np.asarray(lengths)
    largest_cluster = np.argmax(lengths)
    c_mesh.remove_triangles_by_index(
        np.where(clusters != largest_cluster)[0]
    )
    c_mesh = c_mesh.remove_unreferenced_vertices()

    if merge_vertices_relative_scale is not None and merge_vertices_relative_scale > 0:
        vertices = np.asarray(c_mesh.vertices) 
        scale = (vertices.max(axis=0) - vertices.min(axis=0)).min()
        c_mesh = c_mesh.merge_close_vertices(merge_vertices_relative_scale * scale)
        c_mesh.remove_degenerate_triangles()
        c_mesh.remove_duplicated_triangles()
        c_mesh.remove_unreferenced_vertices()
        c_mesh.remove_non_manifold_edges()
        c_mesh.compute_vertex_normals()

    
    if hole_filling_num_neighbors is not None and hole_filling_num_neighbors > 0:
        try:
            import pymeshfix
            with _suppress_c_output():
                mfix = pymeshfix.PyTMesh()
                mfix.load_array(np.asarray(c_mesh.vertices), np.asarray(c_mesh.triangles))
                #mfix.clean()
                mfix.fill_small_boundaries(hole_filling_num_neighbors, refine=False)
            vertices, triangles = mfix.return_arrays()
            c_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(triangles))
        except:
            warnings.warn("pymeshfix is not installed, skipping hole filling")
            
    if taubin_smoothing_iterations > 0:
        c_mesh = c_mesh.filter_smooth_taubin(number_of_iterations=taubin_smoothing_iterations)
    return c_mesh