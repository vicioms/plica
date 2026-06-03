import numpy as np
import pandas as pd
import json
from pathlib import Path
import open3d as o3d
from typing import Optional, Dict, Any, List, Union

def _to_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    if pd.isna(x):
        return default
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y", "t"}
    return bool(x)
def _get_first(row, keys, default=None):
    for key in keys:
        if key in row:
            val = row[key]
            if not pd.isna(val):
                return val
    return default


def load_mesh_base(
    name,
    file,
    scale=1.0,
    flip0=False,
    flip1=False,
    flip2=False,
):

    mesh = o3d.io.read_triangle_mesh(str(file))

    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh '{name}' from file: {file}")

    vertices = np.asarray(mesh.vertices)

    flip0 = _to_bool(flip0)
    flip1 = _to_bool(flip1)
    flip2 = _to_bool(flip2)

    if flip0 or flip1 or flip2:
        vertices = vertices.copy()

        if flip0:
            mn, mx = vertices[:, 0].min(), vertices[:, 0].max()
            vertices[:, 0] = (mx + mn) - vertices[:, 0]

        if flip1:
            mn, mx = vertices[:, 1].min(), vertices[:, 1].max()
            vertices[:, 1] = (mx + mn) - vertices[:, 1]

        if flip2:
            mn, mx = vertices[:, 2].min(), vertices[:, 2].max()
            vertices[:, 2] = (mx + mn) - vertices[:, 2]

        if (int(flip0) + int(flip1) + int(flip2)) % 2 == 1:
            tris = np.asarray(mesh.triangles)
            mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])

        mesh.vertices = o3d.utility.Vector3dVector(vertices)

    scale = 1.0 if pd.isna(scale) else float(scale)

    if scale != 1.0:
        mesh.scale(scale, center=(0, 0, 0))

    mesh.compute_vertex_normals()

    return mesh
def load_mesh_preprocess(
    name,
    file,
    scale=1.0,
    flip0=False,
    flip1=False,
    flip2=False,
    target_num_triangles : Optional[int] = None,
    orient_by_chull : bool = False,
    merge_relative_scale : Optional[float] = None,
    holes_nbe : Optional[int] = None):
    mesh = load_mesh_base(name, file, scale, flip0, flip1, flip2)

    return mesh
    
def from_csv_database(
    file_path_or_dataframe,
    load_mesh_func,
    name_col: str = "name",
    file_col: str = "file",
    tag_cols : Optional[List[str]] = None,
    params_cols : Optional[List[str]] = None,
    root_path : Optional[Union[Path, str]] = None,
    *load_args):
    tag_cols = [] if tag_cols is None else tag_cols
    params_cols = [] if params_cols is None else params_cols
    if isinstance(file_path_or_dataframe, pd.DataFrame):
        mesh_db = file_path_or_dataframe.copy()
    else:
        mesh_db = pd.read_csv(file_path_or_dataframe)

    if name_col not in mesh_db.columns or file_col not in mesh_db.columns:
        raise ValueError(
            f"CSV file must contain columns '{name_col}' and '{file_col}'"
        )

    loaded_meshes = {}

    for _, row in mesh_db.iterrows():
        name = row[name_col]

        if name in loaded_meshes:
            raise ValueError(f"Duplicate mesh name found: {name}")

        file = Path(row[file_col])
        if root_path is not None and not file.is_absolute():
            file = Path(root_path) / file

        scale = _get_first(row, ["rescale", "scale"], 1.0)

        flip0 = _to_bool(_get_first(row, ["flipx", "flip0"], False))
        flip1 = _to_bool(_get_first(row, ["flipy", "flip1"], False))
        flip2 = _to_bool(_get_first(row, ["flipz", "flip2"], False))

        tags = {
            col: row[col]
            for col in tag_cols
            if col in mesh_db.columns and not pd.isna(row[col])
        }

        params = {
            col: row[col]
            for col in params_cols
            if col in mesh_db.columns and not pd.isna(row[col])
        }

        kwargs = {
            "name": name,
            "file": file,
            "scale": scale,
            "flip0": flip0,
            "flip1": flip1,
            "flip2": flip2,
        }

        kwargs |= params

        loaded_meshes[name] = {'mesh': load_mesh_func(*load_args, **kwargs), 'tags': tags}

    return loaded_meshes

def from_json_database(
    file_path,
    load_mesh_func,
    name_key: str = "name",
    file_key: str = "file",
    tag_keys : Optional[List[str]] = None,
    params_keys : Optional[List[str]] = None,
    root_path : Optional[Union[Path, str]] = None,
    *load_args,
):
    tag_keys = [] if tag_keys is None else tag_keys
    params_keys = [] if params_keys is None else params_keys

    with open(file_path, "r") as f:
        mesh_db = json.load(f)

    if not isinstance(mesh_db, list):
        raise ValueError("JSON database must be a list of entries")

    if len(mesh_db) == 0:
        return {}

    loaded_meshes = {}

    for entry in mesh_db:
        if name_key not in entry or file_key not in entry:
            raise ValueError(
                f"Each JSON entry must contain keys '{name_key}' and '{file_key}'"
            )

        name = entry[name_key]

        if name in loaded_meshes:
            raise ValueError(f"Duplicate mesh name found: {name}")

        file = Path(entry[file_key])
        if root_path is not None and not file.is_absolute():
            file = Path(root_path) / file

        scale = entry.get("rescale", entry.get("scale", 1.0))

        flip0 = _to_bool(_get_first(entry, ["flipx", "flip0"], False))
        flip1 = _to_bool(_get_first(entry, ["flipy", "flip1"], False))
        flip2 = _to_bool(_get_first(entry, ["flipz", "flip2"], False))

        tags = {
            key: entry[key]
            for key in tag_keys
            if key in entry and entry[key] is not None
        }

        params = {
            key: entry[key]
            for key in params_keys
            if key in entry and entry[key] is not None
        }

        kwargs = {
            "name": name,
            "file": file,
            "scale": scale,
            "flip0": flip0,
            "flip1": flip1,
            "flip2": flip2,
        }

        kwargs |= params

        loaded_meshes[name] = {'mesh': load_mesh_func(*load_args, **kwargs), 'tags': tags}

    return loaded_meshes