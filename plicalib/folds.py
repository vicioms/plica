import numpy as np
from numpy.typing import NDArray
from typing import Optional
from plicalib.meshes import TriangleMesh
class FoldAnnotation:
    def __init__(self, vertices : NDArray, triangles : NDArray, vertex_indices : NDArray):
        
        self.mesh = TriangleMesh(vertices, triangles)
        self.vertex_indices = vertex_indices