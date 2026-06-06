import numpy as np
from plicalib import plica_utils

class TriangularMesh:
    def __init__(self, vertices, triangles):
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Vertices must be a 2D array with shape (n_vertices, 3)")
        self.vertices = vertices
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Triangles must be a 2D array with shape (n_triangles, 3)")
        self.triangles = triangles

        # properties computed and cached
        self._vertex_normals = None
        self._face_normals = None
        self._vertices_adj_matr = None
        self._faces_adj_matr = None
        self._cotan_weights = None
        self._cotan_matrix = None
        self._mass_matrix = None


    @property
    def vertex_normals(self):
        if self._vertex_normals is None:
            self._vertex_normals = plica_utils.compute_vertex_normals(self.vertices, self.triangles)
        return self._vertex_normals
    
    @property
    def face_normals(self):
        if self._face_normals is None:
            self._face_normals = plica_utils.compute_face_normals(self.vertices, self.triangles)
        return self._face_normals
    
    @property
    def vertices_adj_matr(self):
        if self._vertices_adj_matr is None:
            self._vertices_adj_matr = plica_utils.compute_vertices_adjacency_matrix(self.vertices, self.triangles)
        return self._vertices_adj_matr

    @property
    def faces_adj_matr(self):
        if self._faces_adj_matr is None:
            self._faces_adj_matr = plica_utils.compute_faces_adjacency_matrix(self.vertices, self.triangles)
        return self._faces_adj_matr
    

    @property
    def cotan_weights(self):
        if self._cotan_weights is None:
            self._cotan_weights = plica_utils.compute_cotan_weights(self.vertices,
                                                                    self.triangles,
                                                                    return_as_tuple=True,
                                                                    return_edge_lengths=False)
        return self._cotan_weights
    
    @property
    def cotan_matrix(self):
        if self._cotan_matrix is None:
            if self._cotan_weights is None:
                self._cotan_matrix, self._cotan_weights = plica_utils.compute_cotan_matrix(self.vertices,
                                                                    self.triangles,
                                                                    with_diagonal=True,
                                                                    return_cotan_weights=True)
            else:
                self._cotan_matrix = plica_utils.compute_cotan_matrix(self.vertices,
                                                                    self.triangles,
                                                                    precomputed_cotan_weights=self._cotan_weights,
                                                                    with_diagonal=True,
                                                                    return_cotan_weights=False)
        return self._cotan_matrix
    
    @property
    def mass_matrix(self):
        if self._mass_matrix is None:
            self._mass_matrix = plica_utils.compute_voronoi_mass(self.vertices, self.triangles, self.cotan_weights)
        return self._mass_matrix
        


    