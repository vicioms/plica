import numpy as np
from numpy.typing import NDArray
from typing import Optional

class FoldAnnotation:
    def __init__(self, vertices : NDArray, triangles : NDArray, vertex_indices : NDArray, vertex_mean_curvature : Optional[NDArray] = None):
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Vertices must be a 2D array with shape (n_vertices, 3)")
        self.vertices = vertices
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Triangles must be a 2D array with shape (n_triangles, 3)")
        self.triangles = triangles
        if vertex_indices.ndim != 1:
            raise ValueError("Vertex indices must be a 1D array")
        self.vertex_indices = vertex_indices
        if vertex_mean_curvature is not None:
            if vertex_mean_curvature.ndim != 1:
                raise ValueError("Vertex mean curvature must be a 1D array")
            if vertex_mean_curvature.shape[0] != vertices.shape[0]:
                raise ValueError("Vertex mean curvature must have the same length as the number of vertices")
            self.vertex_mean_curvature = vertex_mean_curvature
        else:
            self.vertex_mean_curvature = None
