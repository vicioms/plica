import numpy as np
from scipy.interpolate import make_splprep, interp1d
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Dict
import scipy.sparse as sparse
from scipy.spatial import KDTree
import plicalib.meshes as meshes
import networkx as nx
from tqdm.auto import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import igl



class CurvatureSegmentation:
    def __init__(self, 
                 initial_params : Dict,
                 vertices : np.ndarray, 
                 triangles : np.ndarray,
                 exclude_boundary_loop : bool = True):
        self.segmentation_params_types = {
                    'min_H': float,
                    'max_H': float,
                    'use_pc2': bool,
                    'pc2_quantile': float,
                    'max_num_clusters': int,
                    'expand_distance': float,
                    'expand_graph_distance': int,
                    'join_method': str,  # 'and' or 'or'
                    }
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_normals = igl.per_vertex_normals(vertices, triangles)
        principal_curvatures = igl.principal_curvature(vertices, triangles)
        self.vertex_pc1_directions, self.vertex_pc2_directions = principal_curvatures[0], principal_curvatures[1]
        self.vertex_pc1_values, self.vertex_pc2_values = principal_curvatures[2], principal_curvatures[3]
        #self.vertex_mean_curvature = (self.vertex_pc1_values + self.vertex_pc2_values) / 2.0
        cotmatrix = igl.cotmatrix(vertices, triangles)
        massmatrix = igl.massmatrix(vertices, triangles, igl.MASSMATRIX_TYPE_VORONOI)

        # extract diagonal and invert
        inv_mass = 1.0 / massmatrix.diagonal()  # (N,)

        # apply as row scaling instead of matrix inverse
        temp_vec = cotmatrix @ vertices              # (N, 3)
        temp_vec *= inv_mass[:, None]                # broadcast over xyz columns

        self.vertex_mean_curvature = np.sum(temp_vec * self.vertex_normals, axis=1)
        self.boundary_loop = igl.boundary_loop(triangles)
        if exclude_boundary_loop:
            self.vertex_mean_curvature[self.boundary_loop] = np.nan
            self.vertex_pc2_values[self.boundary_loop] = np.nan
        self.vertex_adj_list = igl.adjacency_list(triangles)
        self.adj_graph = nx.from_dict_of_lists({i: nbrs for i, nbrs in enumerate(self.vertex_adj_list)})
        self.params = {}
        for param_name, param_type in self.segmentation_params_types.items():
            if param_name in initial_params:
                if not isinstance(initial_params[param_name], param_type):
                    raise ValueError(f"Parameter {param_name} must be of type {param_type}.")
                self.params[param_name] = initial_params[param_name]
            else:
                raise ValueError(f"Missing required parameter: {param_name}")
        self.tree = KDTree(self.vertices)
        self._mean_curvature_mask = None
        self._pc2_mask = None
        self._clusters = None
        self._expanded_clusters = None
        #self._annotations = None

    def update_parameter(self, param_name: str, param_value, invalidate_caches: bool = True) -> bool:
        if param_name not in self.segmentation_params_types:
            raise ValueError(f"Unknown parameter: {param_name}")
        if not isinstance(param_value, self.segmentation_params_types[param_name]):
            raise ValueError(f"Parameter {param_name} must be of type {self.segmentation_params_types[param_name]}.")
        old_value = self.params[param_name]
        self.params[param_name] = param_value
        parameter_changed = old_value != param_value
        if invalidate_caches and parameter_changed:
            if param_name in ['min_H', 'max_H']:
                self._mean_curvature_mask = None
                self._clusters = None
                self._expanded_clusters = None
            if param_name in ['use_pc2', 'pc2_quantile']:
                self._pc2_mask = None
                self._clusters = None
                self._expanded_clusters = None
            if param_name in ['max_num_clusters']:
                self._clusters = None
                self._expanded_clusters = None
            if param_name in ['expand_distance', 'expand_graph_distance', 'join_method']:
                self._expanded_clusters = None
        return parameter_changed

    def _get_mean_curvature_mask(self):
        if self._mean_curvature_mask is None:
            self._mean_curvature_mask = (self.vertex_mean_curvature >= self.params['min_H']) & (self.vertex_mean_curvature <= self.params['max_H'])
        return self._mean_curvature_mask
    def _get_pc2_mask(self):
        if self._pc2_mask is None:
            if self.params['use_pc2']:
                pc2_threshold = np.nanquantile(self.vertex_pc2_values, self.params['pc2_quantile'])
                self._pc2_mask = self.vertex_pc2_values >= pc2_threshold
            else:
                self._pc2_mask = None
        return self._pc2_mask
    
    def _compute_clusters(self):
        if self._clusters is None:
            mean_curvature_mask = self._get_mean_curvature_mask()
            pc2_mask = self._get_pc2_mask()
            if pc2_mask is not None:
                combined_mask = mean_curvature_mask & pc2_mask
            else:
                combined_mask = mean_curvature_mask
            subgraph = self.adj_graph.subgraph(np.argwhere(combined_mask).flatten())
            sorted_components = sorted(list(nx.connected_components(subgraph)), key=lambda x: len(x), reverse=True)
            if self.params['max_num_clusters'] is None or self.params['max_num_clusters'] == 0:
                raise ValueError("Parameter 'max_num_clusters' must be a positive integer.")
            sorted_components = sorted_components[:self.params['max_num_clusters']]
            self._clusters = [ np.array(list(comp)) for comp in  sorted_components ]
        return self._clusters
    
    @staticmethod
    def _expand_nodes(graph : nx.Graph, nodes, dist : int):
        if dist <= 0:
            return set(nodes)
        inflated = set(nodes)
        frontier = set(nodes)
        for _ in range(dist):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(graph.neighbors(node))
            next_frontier -= inflated
            inflated.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return inflated

    def _expand_clusters(self):
        clusters = self._compute_clusters()
        if self._expanded_clusters is None:
            self._expanded_clusters = []
            for cluster in clusters:
                
                grown_cluster_by_distance = None
                grown_cluster_by_graph_distance = None
                if self.params['expand_distance'] > 0:
                    grown_cluster_by_distance = set(cluster)
                    indices = self.tree.query_ball_point(self.vertices[cluster], r=self.params['expand_distance'])
                    for nearby_indices in indices:
                        grown_cluster_by_distance.update(nearby_indices)
                if self.params['expand_graph_distance'] > 0:
                    grown_cluster_by_graph_distance = self._expand_nodes(self.adj_graph, cluster, self.params['expand_graph_distance'])

                if self.params['join_method'] == 'and':
                    if grown_cluster_by_distance is not None and grown_cluster_by_graph_distance is not None:
                        final_cluster = grown_cluster_by_distance.intersection(grown_cluster_by_graph_distance)
                    elif grown_cluster_by_distance is not None:
                        final_cluster = grown_cluster_by_distance
                    elif grown_cluster_by_graph_distance is not None:
                        final_cluster = grown_cluster_by_graph_distance
                    else:
                        final_cluster = set(cluster)
                elif self.params['join_method'] == 'or':
                    final_cluster = set(cluster)
                    if grown_cluster_by_distance is not None:
                        final_cluster.update(grown_cluster_by_distance)
                    if grown_cluster_by_graph_distance is not None:
                        final_cluster.update(grown_cluster_by_graph_distance)
                else:
                    raise ValueError(f"Unknown join_method: {self.params['join_method']}")
                self._expanded_clusters.append(np.array(list(final_cluster)))
            self._expanded_clusters = self._expanded_clusters
        return self._expanded_clusters
    
    def run(self):
        return self._expand_clusters()

    def get_segmentation(self, clusters_annotations : Optional[List[str]] = None, include_geometry :  bool = True, include_curvatures : bool = True):
        if self._clusters is None or self._expanded_clusters is None:
            raise ValueError("No clusters to export. Please run the segmentation first.")
        segmentation_dict = {}
        segmentation_dict['params'] = self.params
        if include_geometry:
            segmentation_dict['vertices'] = self.vertices
            segmentation_dict['triangles'] = self.triangles
            segmentation_dict['vertex_adj_list'] = self.vertex_adj_list
        if include_curvatures:
            segmentation_dict['vertex_mean_curvature'] = self.vertex_mean_curvature
            segmentation_dict['vertex_pc1_directions'] = self.vertex_pc1_directions
            segmentation_dict['vertex_pc2_directions'] = self.vertex_pc2_directions
            segmentation_dict['vertex_pc1_values'] = self.vertex_pc1_values
            segmentation_dict['vertex_pc2_values'] = self.vertex_pc2_values

        if clusters_annotations is not None:
            annotations_dict = {}
            annotation_counter = 0
            for annotation in clusters_annotations:
                name = None
                clusters = None
                if ':' in annotation:
                    sub_annotations = annotation.split(':')
                    if len(sub_annotations) == 2:
                        name, clusters = sub_annotations
                else:
                    clusters = annotation
                if name is None:
                    name = "annotation_" + str(annotation_counter)
                if '+' in clusters:
                    clusters = clusters.split('+')
                    clusters = [int(c) for c in clusters]
                else:
                    clusters = [int(clusters)]
                annotations_dict[name] = np.unique(np.concatenate([self._expanded_clusters[c] for c in clusters if c < len(self._expanded_clusters)]))
                annotation_counter += 1
            segmentation_dict['segmentations'] = annotations_dict
        else:
            segmentation_dict['segmentations'] = {f"cluster_{i}": cluster for i, cluster in enumerate(self._expanded_clusters)}
            #segmentation_dict['annotations'] = clusters_annotations
        return segmentation_dict
class Annotation:
    @staticmethod
    def _align_paths_with_dtw(paths):
        
        # Choose a reference path (first path in this case)
        ref_path = paths[0]

        # List to store aligned paths
        aligned_paths = []

        for path in paths:
            # Compute the DTW distance and the alignment path
            distance, alignment_path = fastdtw(ref_path, path, dist=euclidean)
            # Align path to the reference path
            aligned_path = np.array([path[idx] for idx in list(zip(*alignment_path))[1]])
            aligned_paths.append(aligned_path)
        return aligned_paths
    @staticmethod
    def _resample_path(path, num_points):
        
        original_indices = np.linspace(0, 1, len(path))
        target_indices = np.linspace(0, 1, num_points)

        interpolator = interp1d(original_indices, path, axis=0, kind='cubic', fill_value="extrapolate")
        resampled_path = interpolator(target_indices)
        return resampled_path
    @staticmethod
    def _average_aligned_paths(aligned_paths, num_points):
        resampled_paths = [Annotation._resample_path(path, num_points) for path in aligned_paths]
        stacked_paths = np.stack(resampled_paths)
        avg_path = np.mean(stacked_paths, axis=0)
        err_path = np.sqrt(np.var(stacked_paths, axis=0).sum(axis=-1))
        return avg_path, err_path
    
    @staticmethod
    def _merge_path_djistrka(paths, path_weights, graph, edge_weight_attr='weight'):
        all_nodes = np.concatenate(paths)
        unique_nodes, unique_nodes_counts = np.unique(all_nodes, return_counts=True)
        node_to_index= {node: idx for idx, node in enumerate(unique_nodes)}
        max_count = unique_nodes_counts.max()
        subgraph = nx.subgraph(graph, unique_nodes)

        for u, v, d in subgraph.edges(data=True):
            vote_weight = (unique_nodes_counts[node_to_index[u]] + unique_nodes_counts[node_to_index[v]])/(2*max_count)
            d['combined_' + edge_weight_attr] = d[edge_weight_attr] / vote_weight
        best_path_index = np.argmin(path_weights)
        best_path = paths[best_path_index]
        src, dst = best_path[0], best_path[-1]
        merged_path = nx.dijkstra_path(subgraph, src, dst, weight='combined_' + edge_weight_attr)
        return np.array(merged_path)
    def __init__(self, vertices : NDArray, 
                 triangles : NDArray, 
                 vertex_normals : NDArray, 
                 vertex_indices : NDArray, 
                 curvature : Optional[NDArray] = None,
                 path_quantile_level : float = 0.1,
                 exclude_boundary_vertices : bool = True,
                 use_djistrka_merge : bool = False):
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_normals = vertex_normals
        self.vertex_indices = vertex_indices
        
        if curvature is None:
            curvature = meshes.get_vertex_mean_curvature(vertices, triangles, vertex_normals)
        self.curvature = curvature
        edgelist = meshes.get_edgelist(self.triangles)
        boundary_indices = meshes.get_boundary_vertex_indices_from_edgelist(edgelist)
        edgelist_curvature = np.mean(self.curvature[edgelist], axis=1)
        annotation_edgelist_mask = np.isin(edgelist, self.vertex_indices).all(axis=1)
        annotation_faces_mask = np.isin(self.triangles, self.vertex_indices).all(axis=1)

        self.boundary_indices = boundary_indices
        self.annotation_edgelist = edgelist[annotation_edgelist_mask]
        self.annotation_edgelist_curvature = edgelist_curvature[annotation_edgelist_mask]
        self.annotation_faces = self.triangles[annotation_faces_mask]
        self.annotation_graph = nx.Graph()
        self.annotation_graph.add_edges_from(self.annotation_edgelist, weight=self.annotation_edgelist_curvature)

        self.path_quantile_level = path_quantile_level if path_quantile_level < 0.5  else 1.0 - path_quantile_level
        self.exclude_boundary_vertices = exclude_boundary_vertices
        self.use_djistrka_merge = use_djistrka_merge
        self._start_indices = None
        self._end_indices = None

    def is_connected(self):
        return nx.is_connected(self.annotation_graph)

    def compute_vertex_path_boundaries(self, ):
        '''
            Compute the start and end vertex indices for the fold path based on the principal axis of the vertex coordinates.
        '''
        if self._start_indices is not None and self._end_indices is not None:
            return self._start_indices, self._end_indices
        points = self.vertices[self.vertex_indices]
        points_cov = np.cov(points, rowvar=False)
        vals, vecs = np.linalg.eigh(points_cov)
        principal_axis = vecs[:, -1]
        projected_points = (points - points.mean(axis=0, keepdims=True)) @ principal_axis
        quantiles = np.quantile(projected_points, [self.path_quantile_level, 1.0 - self.path_quantile_level])
        start_indices =  self.vertex_indices[(projected_points < quantiles[0])]
        end_indices = self.vertex_indices[(projected_points > quantiles[1])]
        if self.exclude_boundary_vertices:
            start_indices = np.setdiff1d(start_indices, self.boundary_indices)
            end_indices = np.setdiff1d(end_indices, self.boundary_indices)
        self._start_indices = start_indices
        self._end_indices = end_indices
        return self._start_indices, self._end_indices

    def compute_vertex_path_indices(self):
        '''
            Compute the vertex indices along the fold path using Dijkstra's algorithm on the annotation graph.
        '''
        if self._vpath_indices is not None:
            return self._vpath_indices, self._vpath_indices_weights
        _ = self.compute_vertex_weights()
        subgraph = self.vertex_adj_graph.subgraph(self.indices)
        start_indices, end_indices = self.compute_vpath_boundary_indices()
        paths = []
        path_weights = []
        for v1 in start_indices:
            distances_1, paths_1 = nx.algorithms.single_source_dijkstra(subgraph, v1, weight='weight')
            for v2 in end_indices:
                if(v2  in paths_1):
                    path = paths_1[v2]
                    dist = distances_1[v2]
                    if not np.isfinite(dist):
                        continue
                    paths.append(np.array(path))
                    path_weights.append(dist)
        self._vpath_indices = paths
        self._vpath_indices_weights = np.array(path_weights)
        if self.path_weight_quantile_level is not None and self.path_weight_quantile_level < 1.0:
            weight_threshold = np.quantile(self._vpath_indices_weights, self.path_weight_quantile_level)
            valid_paths_mask = self._vpath_indices_weights <= weight_threshold
            self._vpath_indices = [path for i, path in enumerate(self._vpath_indices) if valid_paths_mask[i]]
            self._vpath_indices_weights = self._vpath_indices_weights[valid_paths_mask]
        return self._vpath_indices, self._vpath_indices_weights

    def compute_vpaths(self) -> List[NDArray[np.float64]]:
        '''
            For each path, we return the 3D coordinates of the vertices along the path. 
        '''
        if self._vpaths is not None:
            return self._vpaths
        paths, _ = self.compute_vpath_indices()
        path_vertices = [self.vertices[path] for path in paths]
        self._vpaths = path_vertices
        return self._vpaths
    
        
    def merge_vpaths(self, num_points : Optional[int] = None) -> Tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
        '''
            We merge the paths found by 'compute_vpath_indices' into a single path, by first aligning them using Dynamic Time Warping (DTW) and then averaging the aligned paths. 
            We return the indices of the vertices along the merged path, the 3D coordinates of the vertices along the merged path and the error estimate for each point of the merged path.
        '''

        if self.use_djistrka_merge:
            if self._merged_vpath_indices is not None and self._merged_vpath is not None and self._merged_vpath_error is not None:
                return self._merged_vpath_indices, self._merged_vpath, self._merged_vpath_error
            path_indices, path_weights = self.compute_vpath_indices()
            merged_path_indices = self._merge_path_djistrka(path_indices, path_weights, self.vertex_adj_graph, edge_weight_attr='weight')
            merged_path = self.vertices[merged_path_indices]
            self._merged_vpath_indices = merged_path_indices
            self._merged_vpath = merged_path
            self._merged_vpath_error = np.zeros(len(merged_path))
        else:
            if num_points is None:
                estimated_spacing = 2*self.estimate_typical_edge_length()
                num_points = self.estimate_num_points_for_merged_vpaths(estimated_spacing)
            if self._merged_vpath is not None and self._merged_vpath_error is not None:
                if len(self._merged_vpath) == num_points:
                    return self._merged_vpath_indices, self._merged_vpath, self._merged_vpath_error

            aligned_paths = self._align_paths_with_dtw(self.compute_vpaths())
            merged_path, merged_path_error = self._average_aligned_paths(aligned_paths, num_points=num_points)
            if self.path_inside_segmentation:
                _, sub_indices = KDTree(self.vertices[self.indices]).query(merged_path)
            else:
                _, sub_indices = KDTree(self.vertices).query(merged_path)
            self._merged_vpath_indices = self.indices[sub_indices] if self.path_inside_segmentation else sub_indices
            self._merged_vpath = self.vertices[self._merged_vpath_indices]
            self._merged_vpath_error = merged_path_error
        return self._merged_vpath_indices, self._merged_vpath, self._merged_vpath_error

    def construct_fold_cross_sections_via_spline(
        self,
        spline_smoothing_factor: float,
        verbose: bool = False,
        plane_mesh_slice_epsilon : float = 0.0,
        num_points : Optional[int] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[int, List[NDArray[np.float64]]]]:

        merged_path_indices, merged_path, merged_path_error = self.merge_vpaths(num_points=num_points)
        path_spline, u = make_splprep(merged_path.T, s=spline_smoothing_factor)
        curve = np.ascontiguousarray(np.stack(path_spline(u, nu=0)).T, dtype=np.float64)
        curve_der = np.ascontiguousarray(np.stack(path_spline(u, nu=1)).T, dtype=np.float64)
        curve_der2 = np.ascontiguousarray(np.stack(path_spline(u, nu=2)).T, dtype=np.float64)
        curve_tangents = curve_der / np.linalg.norm(curve_der, axis=-1, keepdims=True)
        # Frenet normal
        curve_normals = curve_der2 - (curve_der2 * curve_tangents).sum(-1, keepdims=True) * curve_tangents  # remove tangent component
        curve_normals /= np.linalg.norm(curve_normals, axis=-1, keepdims=True)
        cross_sections = {}
        cross_sections_normals = {}

        #for i in tqdm(range(len(curve)), desc="Slicing mesh with planes", disable=not verbose):
        #    p0 = curve[i]
        #    n = curve_tangents[i]
        #    segments, segment_tri_indices, segment_edges = plane_mesh_slice(
        #        self.vertices, self.triangles, plane_origin=p0, plane_normal=n, epsilon=plane_mesh_slice_epsilon
        #    )
        #    tri_index_to_segment_index = {tri_idx: seg_idx for seg_idx, tri_idx in enumerate(segment_tri_indices)}
        #    segments_tri_subgraph = self.tri_adj_graph.subgraph(segment_tri_indices).copy()
        #    non_adjacent_segments_groups = list(nx.connected_components(segments_tri_subgraph))
#
        #    cross_sections[i] = []
        #    cross_sections_normals[i] = []
        #    for segments_group in non_adjacent_segments_groups:
        #        longest_shortest_path = get_approx_longest_shortest_path(segments_tri_subgraph.subgraph(segments_group))
        #        #longest_shortest_path = get_longest_shortest_path(segments_tri_subgraph.subgraph(segments_group))
        #        longest_shortest_path_normals = self.triangle_normals[longest_shortest_path]
        #        longest_shortest_path_seg_indices = np.array([tri_index_to_segment_index[tri_idx] for tri_idx in longest_shortest_path])
        #        ordered_points = (segments[longest_shortest_path_seg_indices, 1, :] + segments[longest_shortest_path_seg_indices, 0, :]) / 2
        #        cross_sections[i].append(ordered_points)
        #        cross_sections_normals[i].append(longest_shortest_path_normals)


        

        for i in tqdm(range(len(curve)), desc="Slicing mesh with planes", disable=not verbose):
            p0 = curve[i]
            n = curve_tangents[i]
            paths, tri_paths = meshes.plane_slice_paths(
                self.vertices, self.triangles, plane_origin=p0, plane_normal=n, epsilon=plane_mesh_slice_epsilon, close_loops=True)
            
            #segments, segment_tri_indices, segment_edges = plane_slice(
            #    self.vertices, self.triangles, plane_origin=p0, plane_normal=n, epsilon=plane_mesh_slice_epsilon
            #)
#
            #seg_index_of = {tri: i for i, tri in enumerate(segment_tri_indices)}
            #segments_tri_subgraph = self.tri_adj_graph.subgraph(segment_tri_indices).copy()
            #non_adjacent_segments_groups = list(nx.connected_components(segments_tri_subgraph))
#
            #cross_sections[i] = []
            #cross_sections_normals[i] = []
            #for group in non_adjacent_segments_groups:
            #    tri_path, seg_indices = get_segment_path_from_triangle_path(segments_tri_subgraph, group, seg_index_of)
            #    if tri_path is None:
            #        continue
            #    ordered_points = (segments[seg_indices, 0] + segments[seg_indices, 1]) / 2
            #    cross_sections[i].append(ordered_points)
#
            #    cross_sections_tri_normals = self.triangle_normals[seg_indices]
            #    # project normals to plane orthogonal to curve tangent
            #    cross_sections_tri_normals -= (cross_sections_tri_normals * n).sum(axis=-1, keepdims=True) * n
            #    cross_sections_tri_normals /= np.linalg.norm(cross_sections_tri_normals, axis=-1, keepdims=True)
#
            #    cross_sections_normals[i].append(cross_sections_tri_normals)

        return curve, curve_tangents, curve_normals, cross_sections, cross_sections_normals