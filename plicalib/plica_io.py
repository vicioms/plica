import json
import inspect
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd
import open3d as o3d


# =============================================================================
# Small utilities
# =============================================================================

def _is_missing(x):
    if x is None:
        return True

    try:
        y = pd.isna(x)
    except Exception:
        return False

    if isinstance(y, (bool, np.bool_)):
        return bool(y)

    return False


def _to_bool(x, default=False):
    if _is_missing(x):
        return default

    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y", "t"}

    return bool(x)


def _get_first(row, keys, default=None):
    for key in keys:
        if key in row:
            val = row[key]
            if not _is_missing(val):
                return val
    return default


def _safe_call_load_mesh_func(
    load_mesh_func,
    load_args,
    kwargs,
    mesh_name=None,
    warn_unknown=True,
):
    """
    Calls load_mesh_func(*load_args, **kwargs), but safely.

    If load_mesh_func does not accept **kwargs, unknown keyword arguments are
    dropped and a warning is issued instead of crashing.
    """
    try:
        sig = inspect.signature(load_mesh_func)
    except (TypeError, ValueError):
        return load_mesh_func(*load_args, **kwargs)

    params = sig.parameters

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    if accepts_var_kwargs:
        return load_mesh_func(*load_args, **kwargs)

    allowed_kwargs = {
        name
        for name, p in params.items()
        if p.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }

    unknown_kwargs = {
        key: val
        for key, val in kwargs.items()
        if key not in allowed_kwargs
    }

    filtered_kwargs = {
        key: val
        for key, val in kwargs.items()
        if key in allowed_kwargs
    }

    if warn_unknown and len(unknown_kwargs) > 0:
        where = "" if mesh_name is None else f" for mesh '{mesh_name}'"
        func_name = getattr(load_mesh_func, "__name__", repr(load_mesh_func))

        warnings.warn(
            f"Ignoring unknown keyword argument(s){where} when calling "
            f"{func_name}: {sorted(unknown_kwargs.keys())}",
            UserWarning,
            stacklevel=2,
        )

    return load_mesh_func(*load_args, **filtered_kwargs)


# =============================================================================
# Mesh loading
# =============================================================================

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

    scale = 1.0 if _is_missing(scale) else float(scale)

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
    target_num_triangles: Optional[int] = None,
    orient_by_chull: bool = False,
    merge_relative_scale: Optional[float] = None,
    holes_nbe: Optional[int] = None,
):
    mesh = load_mesh_base(
        name=name,
        file=file,
        scale=scale,
        flip0=flip0,
        flip1=flip1,
        flip2=flip2,
    )

    # Add preprocessing here later.
    #
    # if target_num_triangles is not None:
    #     ...
    #
    # if orient_by_chull:
    #     ...
    #
    # if merge_relative_scale is not None:
    #     ...
    #
    # if holes_nbe is not None:
    #     ...

    return mesh


# =============================================================================
# CSV database
# =============================================================================

def from_csv_database(
    file_path_or_dataframe,
    separate_into_vertices_and_faces: bool,
    csv_sep: str,
    load_mesh_func=load_mesh_preprocess,
    name_col: str = "name",
    file_col: str = "file",
    tag_cols: Optional[List[str]] = None,
    params_cols: Optional[List[str]] = None,
    root_path: Optional[Union[Path, str]] = None,
    *load_args,
    load_params: Optional[Dict[str, Any]] = None,
    **load_overrides,
):
    """
    Load meshes from a CSV database.

    Automatically handled columns:
        name_col
        file_col / path
        scale / rescale
        flipx / flip0
        flipy / flip1
        flipz / flip2

    params_cols:
        Only these extra CSV columns are fetched and passed to load_mesh_func.

    load_params:
        Manual keyword arguments passed to every mesh.

    Direct keyword overrides:
        You can also pass manual parameters directly as kwargs.

    Precedence:
        base args < database params from params_cols < load_params < direct overrides
    """
    tag_cols = [] if tag_cols is None else tag_cols
    params_cols = [] if params_cols is None else params_cols
    load_params = {} if load_params is None else dict(load_params)

    if isinstance(file_path_or_dataframe, pd.DataFrame):
        mesh_db = file_path_or_dataframe.copy()
    else:
        mesh_db = pd.read_csv(file_path_or_dataframe, sep=csv_sep)

    if name_col not in mesh_db.columns:
        raise ValueError(f"CSV file must contain column '{name_col}'")

    if file_col not in mesh_db.columns and "path" not in mesh_db.columns:
        raise ValueError(
            f"CSV file must contain either column '{file_col}' or 'path'"
        )

    loaded_meshes = {}

    for _, row in mesh_db.iterrows():
        name = row[name_col]

        if name in loaded_meshes:
            raise ValueError(f"Duplicate mesh name found: {name}")

        file_value = _get_first(row, [file_col, "path"], None)

        if file_value is None:
            raise ValueError(
                f"Entry for mesh '{name}' must contain either '{file_col}' or 'path'"
            )

        file = Path(file_value)

        if root_path is not None and not file.is_absolute():
            file = Path(root_path) / file

        scale = _get_first(row, ["rescale", "scale"], 1.0)

        flip0 = _to_bool(_get_first(row, ["flipx", "flip0"], False))
        flip1 = _to_bool(_get_first(row, ["flipy", "flip1"], False))
        flip2 = _to_bool(_get_first(row, ["flipz", "flip2"], False))

        tags = {
            col: row[col]
            for col in tag_cols
            if col in mesh_db.columns and not _is_missing(row[col])
        }

        database_params = {
            col: row[col]
            for col in params_cols
            if col in mesh_db.columns and not _is_missing(row[col])
        }

        kwargs = {
            "name": name,
            "file": file,
            "scale": scale,
            "flip0": flip0,
            "flip1": flip1,
            "flip2": flip2,
        }

        kwargs |= database_params
        kwargs |= load_params
        kwargs |= load_overrides

        mesh = _safe_call_load_mesh_func(
            load_mesh_func=load_mesh_func,
            load_args=load_args,
            kwargs=kwargs,
            mesh_name=name,
            warn_unknown=True,
        )

        if separate_into_vertices_and_faces:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            loaded_meshes[name] = {
                "vertices": vertices,
                "triangles": triangles,
                "tags": tags,
            }
        else:
            loaded_meshes[name] = {
                "mesh": mesh,
                "tags": tags,
            }

    return loaded_meshes


# =============================================================================
# JSON database
# =============================================================================

def from_json_database(
    file_path,
    separate_into_vertices_and_faces: bool,
    load_mesh_func=load_mesh_preprocess,
    name_key: str = "name",
    file_key: str = "file",
    tag_keys: Optional[List[str]] = None,
    params_keys: Optional[List[str]] = None,
    root_path: Optional[Union[Path, str]] = None,
    *load_args,
    load_params: Optional[Dict[str, Any]] = None,
    **load_overrides,
):
    """
    Load meshes from a JSON database.

    Supports both list format:

        [
            {
                "name": "mesh_a",
                "file": "mesh_a.ply",
                "scale": 0.5
            }
        ]

    and dictionary format:

        {
            "mesh_a": {
                "path": "mesh_a.ply",
                "scale": 0.5
            }
        }

    Automatically handled fields:
        name_key, or dictionary key as name
        file_key / path
        scale / rescale
        flipx / flip0
        flipy / flip1
        flipz / flip2

    params_keys:
        Only these extra JSON fields are fetched and passed to load_mesh_func.

    load_params:
        Manual keyword arguments passed to every mesh.

    Direct keyword overrides:
        You can also pass manual parameters directly as kwargs.

    Precedence:
        base args < database params from params_keys < load_params < direct overrides
    """
    tag_keys = [] if tag_keys is None else tag_keys
    params_keys = [] if params_keys is None else params_keys
    load_params = {} if load_params is None else dict(load_params)

    with open(file_path, "r") as f:
        mesh_db = json.load(f)

    if isinstance(mesh_db, list):
        entries = []

        for entry in mesh_db:
            if not isinstance(entry, dict):
                raise ValueError("Each JSON list entry must be a dictionary")

            if name_key not in entry:
                raise ValueError(
                    f"Each JSON entry must contain key '{name_key}' "
                    f"when using list format"
                )

            name = entry[name_key]
            entries.append((name, entry))

    elif isinstance(mesh_db, dict):
        entries = list(mesh_db.items())

    else:
        raise ValueError("JSON database must be either a list or a dictionary")

    if len(entries) == 0:
        return {}

    loaded_meshes = {}

    for name, entry in entries:
        if name in loaded_meshes:
            raise ValueError(f"Duplicate mesh name found: {name}")

        if not isinstance(entry, dict):
            raise ValueError(f"Entry for mesh '{name}' must be a dictionary")

        file_value = _get_first(entry, [file_key, "path"], None)

        if file_value is None:
            raise ValueError(
                f"Entry for mesh '{name}' must contain either '{file_key}' or 'path'"
            )

        file = Path(file_value)

        if root_path is not None and not file.is_absolute():
            file = Path(root_path) / file

        scale = _get_first(entry, ["rescale", "scale"], 1.0)

        flip0 = _to_bool(_get_first(entry, ["flipx", "flip0"], False))
        flip1 = _to_bool(_get_first(entry, ["flipy", "flip1"], False))
        flip2 = _to_bool(_get_first(entry, ["flipz", "flip2"], False))

        tags = {
            key: entry[key]
            for key in tag_keys
            if key in entry and not _is_missing(entry[key])
        }

        database_params = {
            key: entry[key]
            for key in params_keys
            if key in entry and not _is_missing(entry[key])
        }

        kwargs = {
            "name": name,
            "file": file,
            "scale": scale,
            "flip0": flip0,
            "flip1": flip1,
            "flip2": flip2,
        }

        kwargs |= database_params
        kwargs |= load_params
        kwargs |= load_overrides

        mesh = _safe_call_load_mesh_func(
            load_mesh_func=load_mesh_func,
            load_args=load_args,
            kwargs=kwargs,
            mesh_name=name,
            warn_unknown=True,
        )
        if separate_into_vertices_and_faces:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            mesh = (vertices, triangles)
            
            loaded_meshes[name] = {
                "vertices": vertices,
                "triangles": triangles,
                "tags": tags,
            }
        else:
            loaded_meshes[name] = {
                "mesh": mesh,
                "tags": tags,
            }

    return loaded_meshes