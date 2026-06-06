import numpy as np
from scipy.spatial import ConvexHull, KDTree
from typing import Dict, Optional, Tuple, Union
from numpy.typing import NDArray, ArrayLike
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
from scipy.interpolate import NearestNDInterpolator

def get_edgelist(faces : NDArray) -> NDArray:
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
def get_dcel_incidence(
    faces: NDArray,
) -> tuple[sparse.csr_matrix, NDArray]:
    faces = np.asarray(faces, dtype=np.int64)

    edgelist = get_edgelist(faces)
    face_indices = np.tile(np.arange(faces.shape[0]), 3)

    sorted_edges = np.sort(edgelist, axis=1)

    signs = np.where(edgelist[:, 0] == sorted_edges[:, 0], 1, -1)

    unique_edges, edge_indices = np.unique(
        sorted_edges,
        axis=0,
        return_inverse=True,
    )

    B = sparse.coo_matrix(
        (signs, (edge_indices, face_indices)),
        shape=(len(unique_edges), len(faces)), dtype=np.int64).tocsr()

    return B, unique_edges
    

def get_faces_adjacency_scipy(faces, return_edges=False):
    """
    Vectorized sparse face adjacency.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n_faces, n_faces)
        A[f, g] = 1 if faces f and g share an edge.

    Optionally returns
    -------
    unique_edges : ndarray, shape (n_edges, 2)
        The unique undirected mesh edges.
    B : scipy.sparse.csr_matrix, shape (n_edges, n_faces)
        Edge-face incidence matrix.
    """
    faces = np.asarray(faces, dtype=np.int64)
    n_faces = faces.shape[0]

    # all triangle edges
    edges = get_edgelist(faces)

    # corresponding face index for each edge
    face_ids = np.tile(np.arange(n_faces), 3)

    # undirected edges
    edges = np.sort(edges, axis=1)

    # compress edges to integer edge ids
    unique_edges, edge_ids = np.unique(
        edges,
        axis=0,
        return_inverse=True,
    )

    n_edges = len(unique_edges)

    # edge-face incidence matrix B[e, f] = 1
    data = np.ones(len(edge_ids), dtype=np.uint8)

    B = sparse.coo_matrix(
        (data, (edge_ids, face_ids)),
        shape=(n_edges, n_faces),
    ).tocsr()

    # face adjacency: faces adjacent if they share an edge
    A = B.T @ B

    # remove self-adjacency
    A.setdiag(0)
    A.eliminate_zeros()

    # binarize
    A.data[:] = 1
    A = A.tocsr()

    if return_edges:
        return A, unique_edges, B

    return A
def get_vertices_adjacency_scipy(faces):
    edgelist = get_edgelist(faces)
    n_vertices = edgelist.max() + 1
    data = np.ones(len(edgelist), dtype=np.uint8)
    A = sparse.coo_matrix(
        (data, (edgelist[:, 0], edgelist[:, 1])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A
def get_boundary_vertices(faces: NDArray) -> NDArray:
    edgelist = get_edgelist(faces)
    sorted_edges = np.sort(edgelist, axis=1)
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    return np.unique(boundary_edges.flatten())
def get_vertices_connected_components(vertices_adjacency: sparse.csr_matrix, vertex_indices: NDArray) -> NDArray:
    subgraph = vertices_adjacency[vertex_indices][:, vertex_indices]
    n_components, labels = connected_components(subgraph, directed=False, return_labels=True)
    return labels

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

def compute_cotan_weights(
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
def compute_cotan_matrix(vertices: NDArray, faces: NDArray, precomputed_cotan_weights: Optional[Tuple[NDArray, NDArray, NDArray]] = None, with_diagonal : bool = True, return_cotan_weights: bool = False) -> sparse.csr_matrix:
    if precomputed_cotan_weights is None:
        cotan_0, cotan_1, cotan_2 = compute_cotan_weights(
            vertices,
            faces,
            return_as_tuple=True,
        )
    else:
        cotan_0, cotan_1, cotan_2 = precomputed_cotan_weights

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
    if return_cotan_weights:
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
        cotan_0, cotan_1, cotan_2 = compute_cotan_weights(
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
def get_vertex_mean_curvature(vertices, faces, vertex_normals=None, cotan_matrix=None, mass_matrix=None):
    if vertex_normals is None:
        vertex_normals = compute_vertex_normals(vertices, faces, normalize=True)
    if cotan_matrix is None and mass_matrix is None:
        cotan_matrix, cotan_weights = compute_cotan_matrix(vertices, faces, with_diagonal=True, return_cotan_weights=True)
        mass_matrix = compute_voronoi_mass(vertices, faces, cotans=cotan_weights)
    elif cotan_matrix is None:
        cotan_matrix = compute_cotan_matrix(vertices, faces, with_diagonal=True, return_cotan_weights=False)
    elif mass_matrix is None:
        mass_matrix = compute_voronoi_mass(vertices, faces)
    return 0.5 * (vertex_normals * (cotan_matrix @ vertices)).sum(axis=1) / mass_matrix

def _plane_slice_edge_crosses(si, sj, epsilon=0):
    """
    Return True where an edge crosses the slicing plane.

    Vertices exactly on the plane are treated as positive-side vertices,
    with a small numerical floor when epsilon <= 0.
    """
    eps = max(float(epsilon), 1e-12)
    si = np.where(np.abs(si) <= eps, eps, si)
    sj = np.where(np.abs(sj) <= eps, eps, sj)
    return (si > 0) != (sj > 0)
def plane_slice(
    vertices: NDArray,
    faces: NDArray,
    plane_origin: NDArray,
    plane_normal: NDArray,
    epsilon=0,
    return_crossed_edges=False,
) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
    '''
    Slice a triangular mesh with a plane defined by `plane_origin` and `plane_normal`.

    Parameters
    ----------
    vertices : array, shape (N, 3)
        Vertex coordinates of the mesh.

    faces : array, shape (M, 3)
        Indices of vertices forming triangular faces.

    plane_origin : array, shape (3,)
        A point on the slicing plane.

    plane_normal : array, shape (3,)
        Normal vector of the slicing plane.

    epsilon : float
        Numerical stability threshold for determining if a vertex is on the plane.

    return_crossed_edges : bool
        If True, also return crossed_edge_vertices with shape (K, 2, 2).
        For segment k and endpoint a, crossed_edge_vertices[k, a] contains
        the two original vertex ids of the mesh edge that generated that endpoint.

    Returns
    -------
    segments : array, shape (K, 2, 3)
        Line segments representing the intersection of the mesh with the plane.

    triangle_indices : array, shape (K,)
        Indices of the triangles that were sliced to produce each segment.

    crossed_edge_vertices : array, shape (K, 2, 2), optional
        Crossed mesh edges represented by their endpoint vertex ids.
    '''
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    plane_origin = np.asarray(plane_origin, dtype=vertices.dtype)
    plane_normal = np.asarray(plane_normal, dtype=vertices.dtype)

    signed_distances = (vertices - plane_origin) @ plane_normal

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    s0, s1, s2 = signed_distances[i0], signed_distances[i1], signed_distances[i2]

    c01_crosses = _plane_slice_edge_crosses(s0, s1, epsilon)
    c12_crosses = _plane_slice_edge_crosses(s1, s2, epsilon)
    c20_crosses = _plane_slice_edge_crosses(s2, s0, epsilon)

    den01 = s0 - s1
    den12 = s1 - s2
    den20 = s2 - s0

    t01 = np.full(s0.shape, np.nan, dtype=vertices.dtype)
    t12 = np.full(s0.shape, np.nan, dtype=vertices.dtype)
    t20 = np.full(s0.shape, np.nan, dtype=vertices.dtype)

    t01[c01_crosses] = s0[c01_crosses] / den01[c01_crosses]
    t12[c12_crosses] = s1[c12_crosses] / den12[c12_crosses]
    t20[c20_crosses] = s2[c20_crosses] / den20[c20_crosses]

    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

    p01 = v0 + (v1 - v0) * t01[:, None]
    p12 = v1 + (v2 - v1) * t12[:, None]
    p20 = v2 + (v0 - v2) * t20[:, None]

    P = np.stack([p01, p12, p20], axis=1)
    M = np.stack([c01_crosses, c12_crosses, c20_crosses], axis=1)

    good_triangles = np.sum(M, axis=1) == 2
    triangle_indices = np.flatnonzero(good_triangles)

    P_good = P[good_triangles]
    M_good = M[good_triangles]

    segments = P_good[M_good].reshape(-1, 2, 3)

    if return_crossed_edges:
        # E[f, e] is the original vertex pair of local edge e of face f.
        # The local edge order matches P/M:
        #   edge 0 -> (faces[:, 0], faces[:, 1])
        #   edge 1 -> (faces[:, 1], faces[:, 2])
        #   edge 2 -> (faces[:, 2], faces[:, 0])
        E = np.stack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ],
            axis=1,
        )

        E_good = E[good_triangles]

        # Since each good triangle has exactly two crossed edges,
        # this has shape (num_segments, 2 endpoints, 2 vertex ids per mesh edge).
        crossed_edge_vertices = E_good[M_good].reshape(-1, 2, 2)

        return segments, triangle_indices, crossed_edge_vertices

    return segments, triangle_indices
def plane_slice_single_normal(
    vertices: NDArray,
    triangles: NDArray,
    plane_origins: NDArray,
    plane_normal: NDArray,
    epsilon=0,
    return_crossed_edges=False,
    return_as_dict=False,
):
    v_dot_n = vertices @ plane_normal                     # (num_vertices,)
    o_dot_n = plane_origins @ plane_normal               # (num_planes,)

    # (num_planes, num_vertices)
    signed_distances = v_dot_n[None, :] - o_dot_n[:, None]

    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    # (num_planes, num_triangles)
    s0 = signed_distances[:, i0]
    s1 = signed_distances[:, i1]
    s2 = signed_distances[:, i2]

    c01_crosses = _plane_slice_edge_crosses(s0, s1, epsilon)
    c12_crosses = _plane_slice_edge_crosses(s1, s2, epsilon)
    c20_crosses = _plane_slice_edge_crosses(s2, s0, epsilon)

    den01 = s0 - s1
    den12 = s1 - s2
    den20 = s2 - s0

    t01 = np.full(s0.shape, np.nan, dtype=vertices.dtype)
    t12 = np.full(s0.shape, np.nan, dtype=vertices.dtype)
    t20 = np.full(s0.shape, np.nan, dtype=vertices.dtype)

    t01[c01_crosses] = s0[c01_crosses] / den01[c01_crosses]
    t12[c12_crosses] = s1[c12_crosses] / den12[c12_crosses]
    t20[c20_crosses] = s2[c20_crosses] / den20[c20_crosses]

    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

    p01 = v0[None, :, :] + (v1 - v0)[None, :, :] * t01[:, :, None]
    p12 = v1[None, :, :] + (v2 - v1)[None, :, :] * t12[:, :, None]
    p20 = v2[None, :, :] + (v0 - v2)[None, :, :] * t20[:, :, None]

    P = np.stack([p01, p12, p20], axis=2)
    M = np.stack([c01_crosses, c12_crosses, c20_crosses], axis=2)

    good = np.sum(M, axis=2) == 2
    plane_ids, tri_ids = np.nonzero(good)

    P_good = P[good]
    M_good = M[good]

    segments = P_good[M_good].reshape(-1, 2, 3)

    if return_crossed_edges:
        E = np.stack(
            [
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ],
            axis=1,
        )
        E_good = E[tri_ids]
        crossed_edges = E_good[M_good].reshape(-1, 2, 2)
    else:
        crossed_edges = None

    if return_as_dict:
        results = {}
        for plane_id in range(plane_origins.shape[0]):
            plane_mask = plane_ids == plane_id
            results[plane_id] = {
                'segments': segments[plane_mask],
                'triangle_indices': tri_ids[plane_mask],
            }
            if crossed_edges is not None:
                results[plane_id]['crossed_edges'] = crossed_edges[plane_mask]
        return results

    if crossed_edges is not None:
        return segments, plane_ids, tri_ids, crossed_edges

    return segments, plane_ids, tri_ids
def _order_path_from_csr(A: sparse.csr_matrix, close_loops: bool = True) -> NDArray:
    """
    Order nodes in a path-like sparse graph.

    This assumes each connected component is either an open chain
    or a simple closed loop. For branching/non-manifold components,
    this returns one greedy walk through the component.
    """
    A = A.tocsr()
    n = A.shape[0]

    if n == 0:
        return np.empty(0, dtype=np.int64)

    if n == 1:
        return np.array([0], dtype=np.int64)

    deg = np.diff(A.indptr)
    endpoints = np.flatnonzero(deg == 1)
    is_closed = len(endpoints) == 0

    if len(endpoints) > 0:
        start = endpoints[0]
    else:
        start = 0

    path = []
    visited = np.zeros(n, dtype=bool)

    prev = -1
    cur = int(start)

    while True:
        path.append(cur)
        visited[cur] = True

        neighbors = A.indices[A.indptr[cur]:A.indptr[cur + 1]]
        candidates = neighbors[(neighbors != prev) & (~visited[neighbors])]

        if len(candidates) == 0:
            break

        prev, cur = cur, int(candidates[0])

    path = np.asarray(path, dtype=np.int64)

    if close_loops and is_closed and len(path) > 2:
        path = np.concatenate([path, path[:1]])

    return path
def _segments_to_polylines_from_crossed_edges(
    segments: NDArray,
    crossed_edge_vertices: NDArray,
    tri_indices: Optional[NDArray] = None,
    close_loops: bool = True,
):
    """
    Convert unordered mesh-plane intersection segments into ordered polylines.

    Nodes of the graph are crossed original mesh edges, represented by
    their sorted vertex pairs. Edges of the graph are slice segments.
    This orders actual segment endpoints, not segment midpoints.
    """
    segments = np.asarray(segments)
    crossed_edge_vertices = np.asarray(crossed_edge_vertices, dtype=np.int64)

    n_segments = len(segments)
    if n_segments == 0:
        return [], []

    # Endpoint labels: the original mesh edge on which each endpoint lies.
    endpoint_edges = crossed_edge_vertices.reshape(-1, 2)
    endpoint_edges = np.sort(endpoint_edges, axis=1)

    # Endpoint coordinates in matching order.
    endpoint_points = segments.reshape(-1, 3)

    unique_edges, inv = np.unique(
        endpoint_edges,
        axis=0,
        return_inverse=True,
    )

    n_nodes = len(unique_edges)
    endpoint_node_ids = inv.reshape(n_segments, 2)

    # There may be duplicate coordinates for the same crossed mesh edge
    # from the two adjacent triangles. Average them for numerical robustness.
    node_points = np.zeros((n_nodes, 3), dtype=segments.dtype)
    node_counts = np.zeros(n_nodes, dtype=np.int64)

    np.add.at(node_points, inv, endpoint_points)
    np.add.at(node_counts, inv, 1)
    node_points /= node_counts[:, None]

    u = endpoint_node_ids[:, 0]
    v = endpoint_node_ids[:, 1]

    # Degenerate segments can happen when the plane exactly hits vertices/edges.
    valid = u != v
    valid_seg_ids = np.flatnonzero(valid)
    u = u[valid]
    v = v[valid]

    if len(u) == 0:
        return [], []

    data = np.ones(2 * len(u), dtype=np.uint8)
    A = sparse.coo_matrix(
        (
            data,
            (
                np.concatenate([u, v]),
                np.concatenate([v, u]),
            ),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()

    A.data[:] = 1
    A.eliminate_zeros()

    # Map graph edge -> source slice segment(s).
    # Multiple segments on same pair should only happen in degenerate/non-manifold cases.
    edge_to_seg = {}
    for seg_id in valid_seg_ids:
        a, b = endpoint_node_ids[seg_id]
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        edge_to_seg.setdefault((a, b), []).append(int(seg_id))

    n_components, labels = connected_components(
        A,
        directed=False,
        return_labels=True,
    )

    paths = []
    tri_paths = []

    tri_indices_arr = None if tri_indices is None else np.asarray(tri_indices)

    for c in range(n_components):
        global_nodes = np.flatnonzero(labels == c)

        if len(global_nodes) == 0:
            continue

        A_comp = A[global_nodes][:, global_nodes].tocsr()
        local_order = _order_path_from_csr(A_comp, close_loops=close_loops)
        ordered_nodes = global_nodes[local_order]

        ordered_points = node_points[ordered_nodes]

        ordered_seg_ids = []
        for a, b in zip(ordered_nodes[:-1], ordered_nodes[1:]):
            aa, bb = int(a), int(b)
            if aa > bb:
                aa, bb = bb, aa

            candidates = edge_to_seg.get((aa, bb), [])
            if len(candidates) > 0:
                ordered_seg_ids.append(candidates[0])

        ordered_seg_ids = np.asarray(ordered_seg_ids, dtype=np.int64)

        paths.append(ordered_points)

        if tri_indices_arr is None:
            tri_paths.append(None)
        else:
            tri_paths.append(tri_indices_arr[ordered_seg_ids])

    return paths, tri_paths
def plane_slice_paths(
    vertices,
    faces,
    plane_origin,
    plane_normal,
    epsilon=0,
    close_loops: bool = True,
):
    """
    Slice mesh and return ordered intersection polylines.

    Unlike the previous version, this returns actual ordered intersection
    endpoints, not segment midpoints.

    """
    segments, tri_indices, crossed_edge_vertices = plane_slice(
        vertices,
        faces,
        plane_origin,
        plane_normal,
        epsilon=epsilon,
        return_crossed_edges=True,
    )

    paths, tri_paths = _segments_to_polylines_from_crossed_edges(
        segments,
        crossed_edge_vertices,
        tri_indices=tri_indices,
        close_loops=close_loops,
    )

    return paths, tri_paths


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