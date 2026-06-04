import numpy as np
from scipy.spatial import ConvexHull, KDTree
from typing import Optional, Tuple, Union
from numpy.typing import NDArray, ArrayLike
import scipy.sparse as sparse
from scipy.interpolate import NearestNDInterpolator

def nn_interpolation(vertices: NDArray, values: NDArray, query_points: NDArray) -> NDArray:
    interpolator = NearestNDInterpolator(vertices, values)
    return interpolator(query_points)
def normals_misalignment_chull(points: NDArray, normals: NDArray, eps: float = 1e-12, return_dots: bool = False):
    """
    Compute the ratio of points whose normals are misaligned with the
    direction from the convex-hull centroid to the point.

    Parameters
    ----------
    points : array, shape (N, 3)
        Point coordinates.

    normals : array, shape (N, 3)
        Normal vectors associated with each point.

    eps : float
        Numerical stability threshold.

    return_dots : bool
        If True, also return the dot products.

    Returns
    -------
    misaligned_ratio : float
        Fraction of points with inward-pointing normals.

    dots : array, optional, shape (N,)
        Dot products between outward radial directions and normals.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)

    if points.shape != normals.shape:
        raise ValueError(
            f"`points` and `normals` must have the same shape. "
            f"Got {points.shape} and {normals.shape}."
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"`points` must have shape (N, 3). Got {points.shape}.")

    hull = ConvexHull(points)

    hull_points = points[hull.vertices]
    hull_centroid = hull_points.mean(axis=0)

    radial_dirs = points - hull_centroid[None, :]
    radial_norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)

    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)

    valid = (radial_norms[:, 0] > eps) & (normal_norms[:, 0] > eps)

    radial_dirs_unit = np.zeros_like(radial_dirs)
    normals_unit = np.zeros_like(normals)

    radial_dirs_unit[valid] = radial_dirs[valid] / radial_norms[valid]
    normals_unit[valid] = normals[valid] / normal_norms[valid]

    dots = np.sum(radial_dirs_unit * normals_unit, axis=1)

    misaligned_ratio = np.mean(dots[valid] < 0.0)

    if return_dots:
        return misaligned_ratio, dots

    return misaligned_ratio

# simple geometric utilities for meshes
def compute_face_signed_volumes(vertices: NDArray, faces: NDArray) -> NDArray:
    """
    Compute the contribution of each face to the signed volume enclosed by a mesh defined by vertices and faces.

    Parameters
    ----------
    vertices : array, shape (N, 3)
        Vertex coordinates.

    faces : array, shape (M, 3)
        Indices of vertices forming triangular faces.

    Returns
    -------
    volume : array, shape (M,)
        Contribution of each face to the signed volume enclosed by the mesh.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    return np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
def compute_face_areas(vertices: NDArray, faces: NDArray) -> NDArray:
    """
    Compute the area of each face in a mesh defined by vertices and faces.

    Parameters
    ----------
    vertices : array, shape (N, 3)
        Vertex coordinates.

    faces : array, shape (M, 3)
        Indices of vertices forming triangular faces.

    Returns
    -------
    area : array, shape (M,)
        Area of each triangular face.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    return np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2.0
def compute_face_normals(vertices: NDArray, faces: NDArray, normalize: bool = True, return_norm: bool = False) -> Tuple[NDArray, Optional[NDArray]]:
    """
    Compute the normal vector for each face in a mesh defined by vertices and faces.

    Parameters
    ----------
    vertices : array, shape (N, 3)
        Vertex coordinates.

    faces : array, shape (M, 3)
        Indices of vertices forming triangular faces.

    Returns
    -------
    normals : array, shape (M, 3)
        Normal vector for each triangular face.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    if return_norm or normalize:
         norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if normalize:
        normals = normals/norms
    if return_norm:
        return normals, norms
    return normals
def compute_vertex_normals(vertices: NDArray, faces: NDArray, face_normals: Optional[NDArray] = None, normalize: bool = False, area_weighted: bool = False) -> NDArray:
    """
    Compute the normal vector for each vertex in a mesh defined by vertices and faces.

    Parameters
    ----------
    vertices : array, shape (N, 3)
        Vertex coordinates.

    faces : array, shape (M, 3)
        Indices of vertices forming triangular faces.

    face_normals : array, shape (M, 3), optional
        Precomputed normal vectors for each face. If not provided, they will be computed internally.

    area_weighted : bool
        If True, weight the face normals by the area of the corresponding faces when accumulating vertex normals. This can lead to smoother results, especially for meshes with non-uniform face sizes.

    Returns
    -------
    normals : array, shape (N, 3)
        Normal vector for each vertex.
    """
    vertex_normals = np.zeros_like(vertices)
    if area_weighted or face_normals is None:
        face_normals = compute_face_normals(vertices, faces, normalize=not area_weighted, return_norm=False)
    
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)

    return vertex_normals/np.linalg.norm(vertex_normals, axis=1, keepdims=True) if normalize else vertex_normals

def compute_edge_cotans(edge_0 : NDArray, edge_1 : NDArray) -> NDArray:
    """
    Compute the cotangent of the angle between two edges.

    Parameters
    ----------
    edge_0 : array, shape (N, 3)
        First edge vectors.

    edge_1 : array, shape (N, 3)
        Second edge vectors.

    Returns
    -------
    cotangent : array, shape (N,)
        Cotangent of the angle between the two edges.
    """
    dot = np.einsum("ij,ij->i", edge_0, edge_1)
    cross_norm = np.linalg.norm(np.cross(edge_0, edge_1), axis=1)
    return dot / cross_norm
# generic utilities for meshes
def get_edgelist(faces : NDArray) -> NDArray:
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)

def compute_cotangent_weights(
    vertices: NDArray,
    faces: NDArray,
    return_as_tuple: bool = False,
    return_edge_lengths : bool = False,
) -> NDArray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e_01 = v1 - v0
    e_12 = v2 - v1
    e_20 = v0 - v2

    # angle at v0, opposite edge (v1, v2)
    cotan_0 = compute_edge_cotans(e_01, -e_20)

    # angle at v1, opposite edge (v2, v0)
    cotan_1 = compute_edge_cotans(e_12, -e_01)

    # angle at v2, opposite edge (v0, v1)
    cotan_2 = compute_edge_cotans(e_20, -e_12)

    if return_as_tuple:
        if return_edge_lengths:
            return (cotan_0, cotan_1, cotan_2), (np.linalg.norm(e_01, axis=1), np.linalg.norm(e_12, axis=1), np.linalg.norm(e_20, axis=1))
        return (cotan_0, cotan_1, cotan_2)
    if return_edge_lengths:
        return np.stack([cotan_0, cotan_1, cotan_2], axis=-1), np.stack([np.linalg.norm(e_01, axis=1), np.linalg.norm(e_12, axis=1), np.linalg.norm(e_20, axis=1)], axis=-1)
    return np.stack([cotan_0, cotan_1, cotan_2], axis=-1)
def compute_cotangent_matrix(vertices: NDArray, faces: NDArray, with_diagonal : bool = True, return_cotangent_weights: bool = False) -> sparse.csr_matrix:
    cotan_0, cotan_1, cotan_2 = compute_cotangent_weights(
        vertices,
        faces,
        return_as_tuple=True,
    )

    i0 = faces[:, 0]
    i1 = faces[:, 1]
    i2 = faces[:, 2]

    # cotan_0 belongs to edge (i1, i2)
    # cotan_1 belongs to edge (i2, i0)
    # cotan_2 belongs to edge (i0, i1)

    i = np.concatenate([i1, i2, i0, i2, i0, i1])
    j = np.concatenate([i2, i0, i1, i1, i2, i0])
    v = np.concatenate([cotan_0, cotan_1, cotan_2,
                        cotan_0, cotan_1, cotan_2])

    n = vertices.shape[0]

    cotan_matrix = 0.5*sparse.coo_matrix((v, (i, j)), shape=(n, n)).tocsr()
    if with_diagonal:
        cotan_matrix.setdiag(-cotan_matrix.sum(axis=1).A1)
    if return_cotangent_weights:
        return cotan_matrix, (cotan_0, cotan_1, cotan_2)
    return cotan_matrix

def compute_voronoi_mass(
    vertices: NDArray,
    faces: NDArray,
    cotans: Optional[Union[NDArray, tuple]] = None,
) -> NDArray:
    """
    Mixed Voronoi vertex areas. 
    From "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds" by Barr et al.

    cotans[:, 0] = cot angle at faces[:, 0]
    cotans[:, 1] = cot angle at faces[:, 1]
    cotans[:, 2] = cot angle at faces[:, 2]
    """

    if cotans is None:
        cotan_0, cotan_1, cotan_2 = compute_cotangent_weights(
            vertices,
            faces,
            return_as_tuple=True,
        )
    else:
        if isinstance(cotans, tuple):
            if len(cotans) != 3:
                raise ValueError(f"If `cotans` is a tuple, it must have length 3. Got {len(cotans)}.")
            cotan_0, cotan_1, cotan_2 = cotans
        else:
            cotan_0 = cotans[:, 0]
            cotan_1 = cotans[:, 1]
            cotan_2 = cotans[:, 2]

    i0 = faces[:, 0]
    i1 = faces[:, 1]
    i2 = faces[:, 2]

    l2_01 = np.sum((vertices[i1] - vertices[i0]) ** 2, axis=1)
    l2_12 = np.sum((vertices[i2] - vertices[i1]) ** 2, axis=1)
    l2_20 = np.sum((vertices[i0] - vertices[i2]) ** 2, axis=1)

    # Pure Voronoi / circumcentric contribution per triangle
    m0 = 0.125 * (cotan_1 * l2_20 + cotan_2 * l2_01)
    m1 = 0.125 * (cotan_2 * l2_01 + cotan_0 * l2_12)
    m2 = 0.125 * (cotan_0 * l2_12 + cotan_1 * l2_20)

    # Mixed area correction for obtuse triangles
    obtuse_0 = cotan_0 < 0
    obtuse_1 = cotan_1 < 0
    obtuse_2 = cotan_2 < 0
    obtuse = obtuse_0 | obtuse_1 | obtuse_2

    if np.any(obtuse):
        area = compute_face_areas(vertices, faces)

        # For an obtuse triangle:
        # obtuse vertex gets A/2
        # other two vertices get A/4
        m0 = np.where(obtuse, 0.25 * area, m0)
        m1 = np.where(obtuse, 0.25 * area, m1)
        m2 = np.where(obtuse, 0.25 * area, m2)

        m0 = np.where(obtuse_0, 0.5 * area, m0)
        m1 = np.where(obtuse_1, 0.5 * area, m1)
        m2 = np.where(obtuse_2, 0.5 * area, m2)

    mass = np.zeros(vertices.shape[0], dtype=vertices.dtype)

    np.add.at(mass, i0, m0)
    np.add.at(mass, i1, m1)
    np.add.at(mass, i2, m2)

    return mass